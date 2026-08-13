"""Conditional variational mixture of survival experts.

The model targets p(T, E | X) and intentionally does not model p(X | S).
Training uses outcome-informed posterior responsibilities, while deployment
uses a state-only router.  Survival experts use a piecewise-exponential full
likelihood, so every patient has a well-defined censored-data likelihood.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GaussianStateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, state_dim)
        self.logvar = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        hidden = self.backbone(x)
        mean = self.mean(hidden)
        logvar = self.logvar(hidden).clamp(-8.0, 5.0)
        return mean, logvar

    @staticmethod
    def sample(mean, logvar, stochastic=True):
        if not stochastic:
            return mean
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


class PiecewiseExponentialExpert(nn.Module):
    """Piecewise-constant baseline hazard with a latent-state log-risk."""

    def __init__(self, state_dim, n_intervals):
        super().__init__()
        self.log_baseline_hazard = nn.Parameter(torch.full((n_intervals,), -2.0))
        self.risk = nn.Linear(state_dim, 1, bias=False)

    def log_likelihood(self, state, duration, event, boundaries):
        """Return one full censored-data log likelihood per patient."""
        duration = duration.reshape(-1)
        event = event.reshape(-1)
        widths = boundaries[1:] - boundaries[:-1]
        starts = boundaries[:-1].unsqueeze(0)
        exposure = torch.clamp(duration.unsqueeze(1) - starts, min=0.0)
        exposure = torch.minimum(exposure, widths.unsqueeze(0))

        interval = torch.bucketize(duration, boundaries[1:-1], right=False)
        log_h0 = self.log_baseline_hazard.clamp(-12.0, 8.0)
        eta = self.risk(state).squeeze(1).clamp(-12.0, 12.0)
        cumulative_hazard = torch.sum(
            exposure * torch.exp(log_h0).unsqueeze(0), dim=1
        ) * torch.exp(eta)
        event_log_hazard = log_h0[interval] + eta
        return event * event_log_hazard - cumulative_hazard

    def cumulative_hazard(self, state, horizon, boundaries):
        horizon_tensor = torch.full(
            (len(state),), float(horizon), device=state.device, dtype=state.dtype
        )
        widths = boundaries[1:] - boundaries[:-1]
        exposure = torch.clamp(
            horizon_tensor.unsqueeze(1) - boundaries[:-1].unsqueeze(0), min=0.0
        )
        exposure = torch.minimum(exposure, widths.unsqueeze(0))
        baseline = torch.sum(
            exposure * torch.exp(self.log_baseline_hazard.clamp(-12.0, 8.0)), dim=1
        )
        eta = self.risk(state).squeeze(1).clamp(-12.0, 12.0)
        return baseline * torch.exp(eta)


class ConditionalVariationalSurvivalMoE(nn.Module):
    """Latent-state mixture of R, P and RP survival experts.

    E-step:
        r_ik ∝ pi_ik(S_i) p_k(T_i,E_i | S_i^k)
    M-step / generalized variational update:
        experts minimize responsibility-weighted negative log likelihood;
        router minimizes CE(stopgrad(r_i), pi_i) + KL(pi_i || prior);
        encoders additionally receive beta_state KL(q(S|X) || N(0,I)).
    """

    expert_names = ("R", "P", "RP")

    def __init__(self, rad_dim, path_dim, state_dim=8, hidden_dim=32,
                 n_intervals=8, learning_rate=1e-3, beta_state=1e-3,
                 beta_router_prior=0.1, reliability_prior=(1 / 3, 1 / 3, 1 / 3),
                 max_epochs=300, patience=30, mc_train_samples=1,
                 mc_test_samples=32, device="cuda", random_state=None):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_intervals = int(n_intervals)
        self.learning_rate = float(learning_rate)
        self.beta_state = float(beta_state)
        self.beta_router_prior = float(beta_router_prior)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.mc_train_samples = int(mc_train_samples)
        self.mc_test_samples = int(mc_test_samples)
        self.random_state = random_state
        self.device = torch.device(
            device if str(device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )

        self.encoders = nn.ModuleDict({
            "R": GaussianStateEncoder(rad_dim, hidden_dim, state_dim),
            "P": GaussianStateEncoder(path_dim, hidden_dim, state_dim),
            "RP": GaussianStateEncoder(rad_dim + path_dim, hidden_dim, state_dim),
        })
        self.experts = nn.ModuleDict({
            name: PiecewiseExponentialExpert(state_dim, n_intervals)
            for name in self.expert_names
        })
        self.router = nn.Sequential(
            nn.Linear(3 * state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        prior = torch.as_tensor(reliability_prior, dtype=torch.float32)
        if prior.shape != (3,) or torch.any(prior <= 0):
            raise ValueError("reliability_prior must contain three positive values")
        self.register_buffer("reliability_prior", prior / prior.sum())
        self.to(self.device)

    @staticmethod
    def _kl_standard_normal(mean, logvar):
        return 0.5 * torch.sum(
            mean.square() + torch.exp(logvar) - 1.0 - logvar, dim=1
        )

    def _encode(self, x_rad, x_path, stochastic):
        inputs = {"R": x_rad, "P": x_path, "RP": torch.cat([x_rad, x_path], 1)}
        states, moments = {}, {}
        for name in self.expert_names:
            mean, logvar = self.encoders[name](inputs[name])
            states[name] = self.encoders[name].sample(mean, logvar, stochastic)
            moments[name] = (mean, logvar)
        return states, moments

    def _router_probs(self, states):
        state_all = torch.cat([states[name] for name in self.expert_names], dim=1)
        return torch.softmax(self.router(state_all), dim=1)

    def _expert_log_likelihoods(self, states, duration, event):
        return torch.stack([
            self.experts[name].log_likelihood(
                states[name], duration, event, self.time_boundaries_
            ) for name in self.expert_names
        ], dim=1)

    @staticmethod
    def posterior_responsibilities(gate_probs, expert_log_likelihoods):
        """Exact categorical E-step; returned responsibilities are detached."""
        log_joint = torch.log(gate_probs.clamp_min(1e-8)) + expert_log_likelihoods
        return torch.softmax(log_joint, dim=1).detach()

    def _objective(self, x_rad, x_path, duration, event, stochastic=True):
        states, moments = self._encode(x_rad, x_path, stochastic)
        gate = self._router_probs(states)
        log_likelihood = self._expert_log_likelihoods(states, duration, event)
        responsibility = self.posterior_responsibilities(gate, log_likelihood)

        expert_nll = -torch.mean(torch.sum(responsibility * log_likelihood, dim=1))
        router_ce = -torch.mean(torch.sum(
            responsibility * torch.log(gate.clamp_min(1e-8)), dim=1
        ))
        router_kl = torch.mean(torch.sum(
            gate * (
                torch.log(gate.clamp_min(1e-8))
                - torch.log(self.reliability_prior.clamp_min(1e-8)).unsqueeze(0)
            ), dim=1
        ))
        state_kl = torch.mean(torch.stack([
            self._kl_standard_normal(*moments[name]) for name in self.expert_names
        ], dim=1).sum(dim=1))
        loss = (
            expert_nll + router_ce
            + self.beta_router_prior * router_kl
            + self.beta_state * state_kl
        )
        return loss, {
            "expert_nll": expert_nll,
            "router_ce": router_ce,
            "router_kl": router_kl,
            "state_kl": state_kl,
            "responsibility": responsibility,
            "gate": gate,
        }

    def _make_boundaries(self, duration, event):
        event_times = duration[event.astype(bool)]
        source = event_times if len(event_times) >= self.n_intervals else duration
        internal = np.quantile(
            source, np.linspace(0.0, 1.0, self.n_intervals + 1)[1:-1]
        )
        upper = float(np.max(duration)) * 1.001 + 1e-6
        boundaries = np.unique(np.concatenate([[0.0], internal, [upper]])).astype(np.float32)
        if len(boundaries) != self.n_intervals + 1:
            # Heavy ties can collapse quantile boundaries. Equal-width bins
            # retain the prespecified model dimension and a valid likelihood.
            boundaries = np.linspace(
                0.0, upper, self.n_intervals + 1, dtype=np.float32
            )
        boundary_tensor = torch.as_tensor(boundaries, device=self.device)
        if "time_boundaries_" in self._buffers:
            self.time_boundaries_ = boundary_tensor
        else:
            self.register_buffer("time_boundaries_", boundary_tensor)
        self.risk_horizon_ = float(np.median(duration))

    def fit(self, x_rad, x_path, y, validation_fraction=0.2):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
        x_rad = np.asarray(x_rad, dtype=np.float32)
        x_path = np.asarray(x_path, dtype=np.float32)
        duration = np.ascontiguousarray(y["duration"], dtype=np.float32)
        event = np.ascontiguousarray(y["event"], dtype=np.float32)
        self._make_boundaries(duration, event)

        train_idx, val_idx = train_test_split(
            np.arange(len(duration)), test_size=validation_fraction,
            random_state=self.random_state
        )
        tensors = [torch.as_tensor(v, device=self.device) for v in (
            x_rad, x_path, duration, event
        )]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        best, best_loss, stale = None, math.inf, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.train()
            optimizer.zero_grad()
            losses = []
            for _ in range(self.mc_train_samples):
                loss, _ = self._objective(
                    *[tensor[train_idx] for tensor in tensors], stochastic=True
                )
                losses.append(loss)
            train_loss = torch.stack(losses).mean()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss, parts = self._objective(
                    *[tensor[val_idx] for tensor in tensors], stochastic=False
                )
            value = float(val_loss)
            self.history_.append({
                "epoch": epoch + 1, "train_loss": float(train_loss),
                "val_loss": value, "expert_nll": float(parts["expert_nll"]),
                "router_ce": float(parts["router_ce"]),
                "router_kl": float(parts["router_kl"]),
                "state_kl": float(parts["state_kl"]),
            })
            if value < best_loss - 1e-6:
                best_loss, stale = value, 0
                best = copy.deepcopy(self.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best is None:
            raise RuntimeError("Variational survival mixture produced no checkpoint")
        self.load_state_dict(best)
        # Keep the two assignment distributions separate. Responsibilities use
        # observed outcomes and are available only during training; gate
        # probabilities depend on states alone and are the deployable policy.
        self.eval()
        with torch.no_grad():
            _, diagnostics = self._objective(*tensors, stochastic=False)
        self.training_responsibilities_ = (
            diagnostics["responsibility"].cpu().numpy()
        )
        self.training_gate_probs_ = diagnostics["gate"].cpu().numpy()
        self.assignment_distillation_gap_ = float(np.mean(np.sum(
            self.training_responsibilities_
            * (
                np.log(np.clip(self.training_responsibilities_, 1e-8, None))
                - np.log(np.clip(self.training_gate_probs_, 1e-8, None))
            ),
            axis=1,
        )))
        self.is_fitted_ = True
        return self

    def predict(self, x_rad, x_path, hard=True, mc_samples=None):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Model must be fitted before prediction")
        mc_samples = self.mc_test_samples if mc_samples is None else int(mc_samples)
        x_rad = torch.as_tensor(np.asarray(x_rad, dtype=np.float32), device=self.device)
        x_path = torch.as_tensor(np.asarray(x_path, dtype=np.float32), device=self.device)
        gate_samples, risk_samples = [], []
        self.eval()
        with torch.no_grad():
            for _ in range(mc_samples):
                states, _ = self._encode(x_rad, x_path, stochastic=True)
                gate = self._router_probs(states)
                risks = torch.stack([
                    self.experts[name].cumulative_hazard(
                        states[name], self.risk_horizon_, self.time_boundaries_
                    ) for name in self.expert_names
                ], dim=1)
                gate_samples.append(gate)
                risk_samples.append(risks)
        probs = torch.stack(gate_samples).mean(0)
        expert_risks = torch.stack(risk_samples).mean(0)
        actions = torch.argmax(probs, dim=1)
        if hard:
            risk = expert_risks.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            risk = torch.sum(probs * expert_risks, dim=1)
        uncertainty = torch.stack(gate_samples).var(0, unbiased=False).mean(1)
        return (
            risk.cpu().numpy(), actions.cpu().numpy(), probs.cpu().numpy(),
            uncertainty.cpu().numpy(),
        )


class ConditionalVariationalSurvivalPipeline:
    def __init__(self, model, hard=True):
        self.model = model
        self.hard = hard
        self.radiomics_scaler = StandardScaler()
        self.pathomics_scaler = StandardScaler()

    @staticmethod
    def _array(x):
        return np.ascontiguousarray(
            x.values if hasattr(x, "values") else x, dtype=np.float32
        )

    def fit(self, x_rad, x_path, y):
        x_rad = self.radiomics_scaler.fit_transform(self._array(x_rad))
        x_path = self.pathomics_scaler.fit_transform(self._array(x_path))
        self.model.fit(x_rad, x_path, y)
        return self

    def transform(self, x_rad, x_path):
        x_rad = self.radiomics_scaler.transform(self._array(x_rad))
        x_path = self.pathomics_scaler.transform(self._array(x_path))
        risk, actions, probs, uncertainty = self.model.predict(
            x_rad, x_path, hard=self.hard
        )
        self.actions_, self.probs_, self.uncertainty_ = actions, probs, uncertainty
        return risk
