"""Conditional variational mixture of survival experts.

The model targets p(T, E | X) and intentionally does not model p(X | S).
Training uses outcome-informed posterior responsibilities, while deployment
uses a state-only router. Each expert has a deterministic representation and a
compact Bayesian linear survival head sampled by multi-chain Pyro NUTS. Survival
experts use a piecewise-exponential full likelihood, so every patient has a
well-defined censored-data likelihood.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


class RiskRepresentationEncoder(nn.Module):
    """Deterministic representation; uncertainty lives in the Bayesian head."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        hidden = self.backbone(x)
        return hidden


class PiecewiseExponentialExpert(nn.Module):
    """Feature-dependent log-risk head; baseline is shared by the MoE."""

    def __init__(self, representation_dim):
        super().__init__()
        self.risk_coef = nn.Parameter(torch.zeros(representation_dim))

    def log_risk(self, representation):
        return representation.mv(self.risk_coef)

    def log_likelihood(self, log_risk, duration, event, boundaries,
                       log_baseline_hazard):
        """Return one full censored-data log likelihood per patient."""
        duration = duration.reshape(-1)
        event = event.reshape(-1)
        widths = boundaries[1:] - boundaries[:-1]
        starts = boundaries[:-1].unsqueeze(0)
        exposure = torch.clamp(duration.unsqueeze(1) - starts, min=0.0)
        exposure = torch.minimum(exposure, widths.unsqueeze(0))

        interval = torch.bucketize(duration, boundaries[1:-1], right=False)
        log_h0 = log_baseline_hazard.clamp(-12.0, 8.0)
        eta = log_risk.reshape(-1).clamp(-12.0, 12.0)
        cumulative_hazard = torch.sum(
            exposure * torch.exp(log_h0).unsqueeze(0), dim=1
        ) * torch.exp(eta)
        event_log_hazard = log_h0[interval] + eta
        return event * event_log_hazard - cumulative_hazard

    def cumulative_hazard(self, log_risk, horizon, boundaries,
                          log_baseline_hazard):
        horizon_tensor = torch.full(
            (len(log_risk),), float(horizon), device=log_risk.device,
            dtype=log_risk.dtype
        )
        widths = boundaries[1:] - boundaries[:-1]
        exposure = torch.clamp(
            horizon_tensor.unsqueeze(1) - boundaries[:-1].unsqueeze(0), min=0.0
        )
        exposure = torch.minimum(exposure, widths.unsqueeze(0))
        baseline = torch.sum(
            exposure * torch.exp(log_baseline_hazard.clamp(-12.0, 8.0)), dim=1
        )
        eta = log_risk.reshape(-1).clamp(-12.0, 12.0)
        return baseline * torch.exp(eta)


class ConditionalVariationalSurvivalMoE(nn.Module):
    """Mixture of stochastic log-risk experts for R, P and RP.

    E-step:
        r_ik ∝ pi_ik(S_i) p_k(T_i,E_i | S_i^k)
    M-step / generalized variational update:
        experts minimize responsibility-weighted negative log likelihood;
        router minimizes CE(stopgrad(r_i), pi_i) + KL(pi_i || prior);
        posterior log-risk uncertainty comes from HMC samples of the Bayesian
        head; no freely learned variance or state KL is used.
    """

    expert_names = ("R", "P", "RP")

    def __init__(self, rad_dim, path_dim, state_dim=None, hidden_dim=32,
                 n_intervals=8, learning_rate=1e-3,
                 beta_router_prior=0.1, reliability_prior=(1 / 3, 1 / 3, 1 / 3),
                 max_epochs=300, patience=30, mc_train_samples=1,
                 mc_test_samples=32, mcmc_warmup=100, mcmc_samples=200,
                 mcmc_step_size=0.01, mcmc_leapfrog_steps=10,
                 mcmc_chains=2, mcmc_max_tree_depth=8,
                 oof_folds=5, oof_vi_epochs=100, oof_vi_samples=8,
                 prior_scale=1.0, baseline_prior_scale=2.0,
                 router_refit_epochs=100, bayesian_em_iterations=3,
                 responsibility_tolerance=1e-3,
                 responsibility_temperature=2.0,
                 responsibility_prior_mix=0.05,
                 hmc_target_acceptance=0.8, hmc_min_acceptance=0.1,
                 hmc_max_rhat=1.2, hmc_min_ess=10.0,
                 hmc_severe_predictive_rhat=1.5,
                 verbose=True, log_every=10,
                 device="cuda", random_state=None):
        super().__init__()
        # Retained only as a compatibility attribute. The statistical state is
        # now one scalar log-risk per expert.
        self.state_dim = 1
        self.n_intervals = int(n_intervals)
        self.learning_rate = float(learning_rate)
        self.beta_router_prior = float(beta_router_prior)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.mc_train_samples = int(mc_train_samples)
        self.mc_test_samples = int(mc_test_samples)
        self.mcmc_warmup = int(mcmc_warmup)
        self.mcmc_samples = int(mcmc_samples)
        self.mcmc_step_size = float(mcmc_step_size)
        self.mcmc_leapfrog_steps = int(mcmc_leapfrog_steps)
        self.mcmc_chains = int(mcmc_chains)
        self.mcmc_max_tree_depth = int(mcmc_max_tree_depth)
        self.oof_folds = int(oof_folds)
        self.oof_vi_epochs = int(oof_vi_epochs)
        self.oof_vi_samples = int(oof_vi_samples)
        if self.mcmc_chains < 2 or self.mcmc_samples < 2:
            raise ValueError("MCMC diagnostics require at least two chains and samples")
        if self.oof_folds < 2 or self.oof_vi_epochs < 1 or self.oof_vi_samples < 2:
            raise ValueError("OOF VI requires >=2 folds, >=1 epoch and >=2 samples")
        self.prior_scale = float(prior_scale)
        self.baseline_prior_scale = float(baseline_prior_scale)
        self.router_refit_epochs = int(router_refit_epochs)
        self.bayesian_em_iterations = int(bayesian_em_iterations)
        self.responsibility_tolerance = float(responsibility_tolerance)
        self.responsibility_temperature = float(responsibility_temperature)
        self.responsibility_prior_mix = float(responsibility_prior_mix)
        self.hmc_target_acceptance = float(hmc_target_acceptance)
        self.hmc_min_acceptance = float(hmc_min_acceptance)
        self.hmc_max_rhat = float(hmc_max_rhat)
        self.hmc_min_ess = float(hmc_min_ess)
        self.hmc_severe_predictive_rhat = float(hmc_severe_predictive_rhat)
        if self.bayesian_em_iterations < 1:
            raise ValueError("bayesian_em_iterations must be positive")
        if self.responsibility_temperature < 1.0:
            raise ValueError("responsibility_temperature must be at least one")
        if not 0.0 <= self.responsibility_prior_mix < 1.0:
            raise ValueError("responsibility_prior_mix must be in [0, 1)")
        self.verbose = bool(verbose)
        self.log_every = max(1, int(log_every))
        self.random_state = random_state
        self.device = torch.device(
            device if str(device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )

        self.encoders = nn.ModuleDict({
            "R": RiskRepresentationEncoder(rad_dim, hidden_dim),
            "P": RiskRepresentationEncoder(path_dim, hidden_dim),
            "RP": RiskRepresentationEncoder(rad_dim + path_dim, hidden_dim),
        })
        self.experts = nn.ModuleDict({
            name: PiecewiseExponentialExpert(hidden_dim)
            for name in self.expert_names
        })
        self.shared_log_baseline_hazard = nn.Parameter(
            torch.full((n_intervals,), -2.0)
        )
        self.router = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        prior = torch.as_tensor(reliability_prior, dtype=torch.float32)
        if prior.shape != (3,) or torch.any(prior <= 0):
            raise ValueError("reliability_prior must contain three positive values")
        self.register_buffer("reliability_prior", prior / prior.sum())
        for name in self.expert_names:
            self.register_buffer(
                f"representation_mean_{name}", torch.zeros(hidden_dim)
            )
            self.register_buffer(
                f"representation_scale_{name}", torch.ones(hidden_dim)
            )
        self.representation_normalized_ = False
        self.to(self.device)

    def _normalize_representation(self, name, representation):
        if not self.representation_normalized_:
            return representation
        mean = getattr(self, f"representation_mean_{name}")
        scale = getattr(self, f"representation_scale_{name}")
        return (representation - mean) / scale

    def _encode(self, x_rad, x_path, stochastic=False):
        inputs = {"R": x_rad, "P": x_path, "RP": torch.cat([x_rad, x_path], 1)}
        representations, log_risks = {}, {}
        for name in self.expert_names:
            raw = self.encoders[name](inputs[name])
            representations[name] = self._normalize_representation(name, raw)
            log_risks[name] = self.experts[name].log_risk(representations[name]).unsqueeze(1)
        return log_risks, representations

    def _fit_representation_normalization(self, representations):
        """Standardize H before joint inference with one shared baseline."""
        if self.representation_normalized_:
            raise RuntimeError("Representation normalization is already fitted")
        normalized = {}
        self.representation_normalization_diagnostics_ = {}
        with torch.no_grad():
            for name in self.expert_names:
                raw = representations[name]
                mean = raw.mean(0)
                scale = raw.std(0, unbiased=False).clamp_min(1e-6)
                expert = self.experts[name]
                old_beta = expert.risk_coef.detach().clone()
                removed_risk_offset = torch.dot(mean, old_beta)

                getattr(self, f"representation_mean_{name}").copy_(mean)
                getattr(self, f"representation_scale_{name}").copy_(scale)
                expert.risk_coef.copy_(scale * old_beta)
                normalized[name] = (raw - mean) / scale
                self.representation_normalization_diagnostics_[name] = {
                    "raw_mean_abs_max": float(mean.abs().max()),
                    "raw_scale_min": float(scale.min()),
                    "raw_scale_max": float(scale.max()),
                    "removed_log_risk_offset": float(removed_risk_offset),
                }
        self.representation_normalized_ = True
        return normalized

    def _make_router_state(self, log_risks, uncertainties):
        """Nine interpretable terms: risks, log-variances, disagreements."""
        risks = torch.cat([log_risks[name] for name in self.expert_names], dim=1)
        uncertainty = torch.cat(
            [uncertainties[name] for name in self.expert_names], dim=1
        )
        disagreements = torch.stack([
            torch.abs(risks[:, 0] - risks[:, 1])
            / torch.sqrt(uncertainty[:, 0].square() + uncertainty[:, 1].square() + 1e-8),
            torch.abs(risks[:, 0] - risks[:, 2])
            / torch.sqrt(uncertainty[:, 0].square() + uncertainty[:, 2].square() + 1e-8),
            torch.abs(risks[:, 1] - risks[:, 2])
            / torch.sqrt(uncertainty[:, 1].square() + uncertainty[:, 2].square() + 1e-8),
        ], dim=1)
        return torch.cat([risks, uncertainty, disagreements], dim=1)

    def _router_probs(self, log_risks, uncertainties):
        return torch.softmax(
            self.router(self._make_router_state(log_risks, uncertainties)), dim=1
        )

    def _expert_log_likelihoods(self, states, duration, event):
        return torch.stack([
            self.experts[name].log_likelihood(
                states[name], duration, event, self.time_boundaries_,
                self.shared_log_baseline_hazard
            ) for name in self.expert_names
        ], dim=1)

    @staticmethod
    def posterior_responsibilities(gate_probs, expert_log_likelihoods,
                                   temperature=1.0, prior=None,
                                   prior_mix=0.0):
        """Tempered categorical E-step with optional prior smoothing."""
        log_joint = torch.log(gate_probs.clamp_min(1e-8)) + expert_log_likelihoods
        responsibility = torch.softmax(log_joint / temperature, dim=1)
        if prior_mix:
            if prior is None:
                prior = torch.full_like(responsibility, 1.0 / responsibility.shape[1])
            else:
                prior = prior.reshape(1, -1).to(responsibility)
            responsibility = (
                (1.0 - prior_mix) * responsibility + prior_mix * prior
            )
        return responsibility.detach()

    def _update_responsibilities(self, gate_probs, expert_log_likelihoods):
        return self.posterior_responsibilities(
            gate_probs, expert_log_likelihoods,
            temperature=self.responsibility_temperature,
            prior=self.reliability_prior,
            prior_mix=self.responsibility_prior_mix,
        )

    def _objective(self, x_rad, x_path, duration, event, stochastic=True):
        states, representations = self._encode(x_rad, x_path, stochastic)
        zero_uncertainty = {name: torch.zeros_like(states[name]) + 1.0 for name in self.expert_names}
        gate = self._router_probs(states, zero_uncertainty)
        log_likelihood = self._expert_log_likelihoods(states, duration, event)
        responsibility = self._update_responsibilities(gate, log_likelihood)

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
        loss = (
            expert_nll + router_ce
            + self.beta_router_prior * router_kl
        )
        return loss, {
            "expert_nll": expert_nll,
            "router_ce": router_ce,
            "router_kl": router_kl,
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
        total_time = max(float(np.sum(duration)), 1e-8)
        event_rate = max(float(np.sum(event)) / total_time, 1e-8)
        self.baseline_prior_location_ = float(np.log(event_rate))
        with torch.no_grad():
            self.shared_log_baseline_hazard.fill_(self.baseline_prior_location_)

    def _joint_head_log_posterior(self, theta, representations, duration, event,
                                  responsibilities):
        """Posterior of three beta vectors and one shared baseline hazard."""
        d = next(iter(representations.values())).shape[1]
        betas = {
            name: theta[index * d:(index + 1) * d]
            for index, name in enumerate(self.expert_names)
        }
        raw_log_h0 = theta[3 * d:]
        # Bound only the exponential likelihood computation. The Gaussian
        # prior must act on the unconstrained latent; clamping it here would
        # create flat, improper posterior tails that are hostile to HMC/NUTS.
        log_h0 = raw_log_h0.clamp(-12.0, 8.0)
        widths = self.time_boundaries_[1:] - self.time_boundaries_[:-1]
        exposure = torch.clamp(
            duration[:, None] - self.time_boundaries_[:-1][None, :], min=0.0
        )
        exposure = torch.minimum(exposure, widths[None, :])
        interval = torch.bucketize(
            duration, self.time_boundaries_[1:-1], right=False
        )
        baseline_cumulative = (exposure * torch.exp(log_h0)[None, :]).sum(1)
        weighted_loglik = torch.zeros((), device=theta.device)
        for expert_index, name in enumerate(self.expert_names):
            eta = representations[name].mv(betas[name]).clamp(-12.0, 12.0)
            loglik = (
                event * (log_h0[interval] + eta)
                - baseline_cumulative * torch.exp(eta)
            )
            weighted_loglik = weighted_loglik + (
                responsibilities[:, expert_index] * loglik
            ).sum()
        log_prior_beta = -0.5 * sum(
            beta.square().sum() for beta in betas.values()
        ) / self.prior_scale**2
        log_prior_hazard = (
            -0.5 * (
                raw_log_h0 - self.baseline_prior_location_
            ).square().sum()
            / self.baseline_prior_scale**2
        )
        return weighted_loglik + log_prior_beta + log_prior_hazard

    def _hmc_joint_head(self, representations, duration, event,
                        responsibilities):
        """Jointly sample all heads and the shared baseline using Pyro NUTS.

        Chains run sequentially. This avoids CUDA multiprocessing and model
        pickling issues while retaining independent initialization and NUTS
        adaptation for every chain.
        """
        try:
            import pyro
            from pyro.infer.mcmc import MCMC, NUTS
        except ImportError as error:
            raise ImportError(
                "Strategy 7 requires pyro-ppl==1.9.1. Install the updated "
                "requirements-pancia.txt before fitting."
            ) from error

        initial = torch.cat([
            *[self.experts[name].risk_coef.detach() for name in self.expert_names],
            self.shared_log_baseline_hazard.detach(),
        ]).clone()
        chain_draws, state_change_rates, mean_accept_probs = [], [], []
        final_step_sizes = []
        divergence_counts = []

        def potential_fn(params):
            log_posterior = self._joint_head_log_posterior(
                params["theta"], representations, duration, event,
                responsibilities,
            )
            if not torch.isfinite(log_posterior):
                raise FloatingPointError(
                    "Non-finite joint survival log posterior during Pyro NUTS"
                )
            return -log_posterior

        for chain in range(self.mcmc_chains):
            chain_seed = (0 if self.random_state is None else self.random_state) + chain
            pyro.set_rng_seed(chain_seed)
            chain_initial = initial + 0.01 * torch.randn_like(initial)
            kernel = NUTS(
                potential_fn=potential_fn,
                step_size=self.mcmc_step_size,
                adapt_step_size=True,
                adapt_mass_matrix=True,
                full_mass=False,
                target_accept_prob=self.hmc_target_acceptance,
                max_tree_depth=self.mcmc_max_tree_depth,
            )
            chain_sampling_diagnostics = {}

            def capture_sampling_diagnostics(active_kernel, _, stage, __):
                """Capture the post-warmup statistic hidden by diagnostics()."""
                if stage.startswith("Sample"):
                    chain_sampling_diagnostics["mean_accept_prob"] = float(
                        active_kernel._mean_accept_prob
                    )

            sampler = MCMC(
                kernel,
                warmup_steps=self.mcmc_warmup,
                num_samples=self.mcmc_samples,
                num_chains=1,
                initial_params={"theta": chain_initial},
                hook_fn=capture_sampling_diagnostics,
                disable_progbar=True,
                disable_validation=False,
            )
            sampler.run()
            draws = sampler.get_samples(group_by_chain=False)["theta"].detach()
            if draws.shape != (self.mcmc_samples, initial.numel()):
                raise RuntimeError(
                    f"Unexpected Pyro NUTS draw shape {tuple(draws.shape)}"
                )
            diagnostics = sampler.diagnostics()
            state_change_rate = diagnostics.get("acceptance rate", {})
            if isinstance(state_change_rate, dict):
                state_change_rate = next(
                    iter(state_change_rate.values()), float("nan")
                )
            mean_accept_prob = chain_sampling_diagnostics.get(
                "mean_accept_prob", float("nan")
            )
            divergences = diagnostics.get("divergences", {})
            if isinstance(divergences, dict):
                divergence_count = sum(len(value) for value in divergences.values())
            else:
                divergence_count = len(divergences)
            chain_draws.append(draws)
            state_change_rates.append(float(state_change_rate))
            mean_accept_probs.append(float(mean_accept_prob))
            divergence_counts.append(int(divergence_count))
            final_step_sizes.append(float(kernel.step_size))
        chains = torch.stack(chain_draws).cpu()
        samples = chains.reshape(-1, chains.shape[-1])
        d = next(iter(representations.values())).shape[1]
        self.posterior_samples_ = {
            name: {"beta": samples[:, index * d:(index + 1) * d]}
            for index, name in enumerate(self.expert_names)
        }
        self.posterior_samples_["shared_log_baseline_hazard"] = samples[:, 3 * d:]
        rhat, ess, stationary = self._chain_diagnostics(chains)
        block_diagnostics = {}
        parameter_labels = []
        for expert_index, name in enumerate(self.expert_names):
            block = slice(expert_index * d, (expert_index + 1) * d)
            block_diagnostics[name] = {
                "max_rhat": float(rhat[block].max()),
                "median_rhat": float(rhat[block].median()),
                "min_ess": float(ess[block].min()),
            }
            parameter_labels.extend([f"beta_{name}[{j}]" for j in range(d)])
        baseline_block = slice(3 * d, None)
        block_diagnostics["baseline"] = {
            "max_rhat": float(rhat[baseline_block].max()),
            "median_rhat": float(rhat[baseline_block].median()),
            "min_ess": float(ess[baseline_block].min()),
        }
        parameter_labels.extend([
            f"log_baseline[{j}]" for j in range(self.n_intervals)
        ])
        worst_index = int(torch.argmax(rhat))

        predictive_diagnostics = {}
        for expert_index, name in enumerate(self.expert_names):
            beta_chains = chains[:, :, expert_index * d:(expert_index + 1) * d]
            risk_chains = torch.einsum(
                "nd,csd->csn", representations[name].cpu(), beta_chains
            )
            risk_rhat, risk_ess, risk_stationary = self._chain_diagnostics(
                risk_chains
            )
            informative = ~risk_stationary
            if torch.any(informative):
                max_predictive_rhat = float(risk_rhat[informative].max())
                median_predictive_rhat = float(risk_rhat[informative].median())
                p95_predictive_rhat = float(torch.quantile(
                    risk_rhat[informative], 0.95
                ))
                min_predictive_ess = float(risk_ess[informative].min())
                worst_patient = int(torch.argmax(risk_rhat))
            else:
                max_predictive_rhat = median_predictive_rhat = 1.0
                p95_predictive_rhat = 1.0
                min_predictive_ess = float(
                    self.mcmc_chains * self.mcmc_samples
                )
                worst_patient = -1
            predictive_diagnostics[name] = {
                "max_rhat": max_predictive_rhat,
                "median_rhat": median_predictive_rhat,
                "p95_rhat": p95_predictive_rhat,
                "min_ess": min_predictive_ess,
                "worst_patient_index": worst_patient,
                "stationary_patients": int(risk_stationary.sum()),
            }
        mean_accept_prob = float(np.mean(mean_accept_probs))
        mean_state_change_rate = float(np.mean(state_change_rates))
        self.mcmc_diagnostics_ = {
            "joint": {
            # Retain the old key for consumers of saved result dictionaries,
            # but give both Pyro quantities explicit, unambiguous names.
            "acceptance_rate": mean_accept_prob,
            "mean_accept_probability": mean_accept_prob,
            "state_change_rate": mean_state_change_rate,
            "target_accept_probability": self.hmc_target_acceptance,
            "n_samples": self.mcmc_chains * self.mcmc_samples,
            "n_chains": self.mcmc_chains,
            "sampler": "Pyro NUTS",
            "divergences": int(sum(divergence_counts)),
            "max_rhat": float(rhat.max()),
            "min_ess": float(ess.min()),
            "worst_parameter": parameter_labels[worst_index],
            "block_diagnostics": block_diagnostics,
            "predictive_diagnostics": predictive_diagnostics,
            "final_step_size_min": float(np.min(final_step_sizes)),
            "final_step_size_max": float(np.max(final_step_sizes)),
            }
        }
        max_rhat = float(rhat.max())
        min_ess = float(ess.min())
        predictive_max_rhat = max(
            value["max_rhat"] for value in predictive_diagnostics.values()
        )
        predictive_p95_rhat = max(
            value["p95_rhat"] for value in predictive_diagnostics.values()
        )
        predictive_min_ess = min(
            value["min_ess"] for value in predictive_diagnostics.values()
        )
        severe_parameter_failure = max_rhat > 2.0
        predictive_failure = (
            not np.isfinite(predictive_max_rhat)
            or predictive_p95_rhat > self.hmc_max_rhat
            or predictive_max_rhat > self.hmc_severe_predictive_rhat
            or predictive_min_ess < self.hmc_min_ess
        )
        if (
            not np.isfinite(mean_accept_prob)
            or mean_accept_prob < self.hmc_min_acceptance
            or sum(divergence_counts) > 0
            or torch.any(stationary)
            or not np.isfinite(max_rhat)
            or severe_parameter_failure
            or predictive_failure
        ):
            raise RuntimeError(
                "Joint NUTS failed: mean_accept_probability="
                f"{mean_accept_prob:.3f}, state_change_rate="
                f"{mean_state_change_rate:.3f}, stationary_parameters="
                f"{int(stationary.sum())}, max_Rhat={max_rhat:.3f}, "
                f"min_ESS={min_ess:.1f}, predictive_max_Rhat="
                f"{predictive_max_rhat:.3f}, predictive_min_ESS="
                f"{predictive_min_ess:.1f}, predictive_p95_Rhat="
                f"{predictive_p95_rhat:.3f}, divergences="
                f"{sum(divergence_counts)}. Increase warmup/samples, raise "
                "hmc_target_acceptance, or reparameterize the posterior."
            )
        warnings = []
        if max_rhat > self.hmc_max_rhat or min_ess < self.hmc_min_ess:
            warnings.append(
                "Some individual parameters have not fully mixed, but "
                "patient-level log-risk diagnostics passed."
            )
        if predictive_max_rhat > self.hmc_max_rhat:
            warnings.append(
                "At least one patient has elevated predictive R-hat, while "
                "the predictive 95th percentile passed."
            )
        if warnings:
            self.mcmc_diagnostics_["joint"]["warning"] = " ".join(warnings)

    @staticmethod
    def _chain_diagnostics(chains):
        """Classical R-hat and lag-one ESS along chain/sample axes."""
        n_samples = chains.shape[1]
        chain_means = chains.mean(1)
        between = n_samples * chain_means.var(0, unbiased=True)
        raw_within = chains.var(1, unbiased=True).mean(0)
        stationary = raw_within <= 1e-12
        within = raw_within.clamp_min(1e-12)
        posterior_var = (
            (n_samples - 1) * within / n_samples + between / n_samples
        )
        rhat = torch.sqrt(posterior_var / within)
        centered = chains - chains.mean(1, keepdim=True)
        lag1 = (
            (centered[:, :-1] * centered[:, 1:]).mean((0, 1))
            / centered.square().mean((0, 1)).clamp_min(1e-12)
        ).clamp(-0.99, 0.99)
        ess = chains.shape[0] * n_samples * (1 - lag1) / (1 + lag1)
        rhat[stationary] = torch.inf
        ess[stationary] = 0.0
        return rhat, ess, stationary

    def _posterior_log_risk_summary(self, representations):
        means, stds = {}, {}
        for name in self.expert_names:
            beta = self.posterior_samples_[name]["beta"].to(self.device)
            draws = representations[name].matmul(beta.T)
            means[name] = draws.mean(1, keepdim=True)
            stds[name] = draws.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return means, stds

    @staticmethod
    def probability_diagnostics(probabilities):
        """Summarize whether categorical assignments are truly decisive."""
        probs = np.asarray(probabilities, dtype=np.float64)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
        sorted_probs = np.sort(probs, axis=1)
        return {
            "mean_probabilities": probs.mean(0).tolist(),
            "mean_entropy": float(entropy.mean()),
            "normalized_entropy": float(entropy.mean() / math.log(probs.shape[1])),
            "mean_max_probability": float(probs.max(1).mean()),
            "mean_top_two_margin": float(
                (sorted_probs[:, -1] - sorted_probs[:, -2]).mean()
            ),
        }

    def _posterior_predictive_log_likelihood(self, representations, duration,
                                             event):
        """Log E_posterior[p(T,E|theta)] for every patient and expert."""
        baseline_draws = self.posterior_samples_[
            "shared_log_baseline_hazard"
        ].to(self.device).clamp(-12.0, 8.0)
        widths = self.time_boundaries_[1:] - self.time_boundaries_[:-1]
        exposure = torch.clamp(
            duration[:, None] - self.time_boundaries_[:-1][None, :], min=0.0
        )
        exposure = torch.minimum(exposure, widths[None, :])
        interval = torch.bucketize(
            duration, self.time_boundaries_[1:-1], right=False
        )
        baseline_cumulative = exposure.matmul(torch.exp(baseline_draws).T)
        event_log_baseline = baseline_draws[:, interval].T
        n_draws = baseline_draws.shape[0]
        predictive = []
        for name in self.expert_names:
            beta = self.posterior_samples_[name]["beta"].to(self.device)
            eta = representations[name].matmul(beta.T).clamp(-12.0, 12.0)
            draw_loglik = (
                event[:, None] * (event_log_baseline + eta)
                - baseline_cumulative * torch.exp(eta)
            )
            predictive.append(
                torch.logsumexp(draw_loglik, dim=1) - math.log(n_draws)
            )
        return torch.stack(predictive, dim=1)

    def _log_likelihood_from_theta(self, theta, representations, duration,
                                   event):
        """Patient/expert log likelihood for one flattened head parameter."""
        d = next(iter(representations.values())).shape[1]
        log_h0 = theta[3 * d:].clamp(-12.0, 8.0)
        widths = self.time_boundaries_[1:] - self.time_boundaries_[:-1]
        exposure = torch.clamp(
            duration[:, None] - self.time_boundaries_[:-1][None, :], min=0.0
        )
        exposure = torch.minimum(exposure, widths[None, :])
        interval = torch.bucketize(
            duration, self.time_boundaries_[1:-1], right=False
        )
        baseline_cumulative = (exposure * torch.exp(log_h0)[None, :]).sum(1)
        likelihoods = []
        for expert_index, name in enumerate(self.expert_names):
            beta = theta[expert_index * d:(expert_index + 1) * d]
            eta = representations[name].mv(beta).clamp(-12.0, 12.0)
            likelihoods.append(
                event * (log_h0[interval] + eta)
                - baseline_cumulative * torch.exp(eta)
            )
        return torch.stack(likelihoods, dim=1)

    def _oof_variational_evidence(self, representations, duration, event,
                                  responsibilities):
        """Cross-fitted predictive evidence from lightweight Gaussian VI heads.

        Each patient's state and survival evidence are produced by a head that
        did not use that patient's outcome. Encoders remain fixed: this stage
        specifically removes the flexible RP head's in-sample advantage.
        """
        n = len(duration)
        n_folds = min(self.oof_folds, n)
        if n_folds < 2:
            raise ValueError("At least two patients are required for OOF evidence")
        d = next(iter(representations.values())).shape[1]
        # A neutral initialization is essential: initializing a fold from the
        # full-data head would leak held-out outcomes through its parameters.
        initial = torch.cat([
            torch.zeros(3 * d, device=self.device),
            torch.full(
                (self.n_intervals,), self.baseline_prior_location_,
                device=self.device,
            ),
        ])
        oof_means = {
            name: torch.empty((n, 1), device=self.device)
            for name in self.expert_names
        }
        oof_stds = {
            name: torch.empty((n, 1), device=self.device)
            for name in self.expert_names
        }
        oof_loglik = torch.empty((n, 3), device=self.device)
        fold_records = []
        splitter = KFold(n_splits=n_folds, shuffle=False)
        for fold, (train_np, heldout_np) in enumerate(splitter.split(np.arange(n))):
            train_idx = torch.as_tensor(train_np, device=self.device)
            heldout_idx = torch.as_tensor(heldout_np, device=self.device)
            fold_representations = {
                name: value[train_idx] for name, value in representations.items()
            }
            mu = nn.Parameter(initial.clone())
            rho = nn.Parameter(torch.full_like(initial, -3.0))
            optimizer = torch.optim.Adam(
                (mu, rho), lr=max(self.learning_rate, 1e-3)
            )
            generator = torch.Generator(device=self.device)
            generator.manual_seed(
                (0 if self.random_state is None else self.random_state)
                + 1009 * (fold + 1)
            )
            final_loss = math.nan
            for _ in range(self.oof_vi_epochs):
                scale = F.softplus(rho).clamp_min(1e-5)
                elbo_samples = []
                for _ in range(min(4, self.oof_vi_samples)):
                    epsilon = torch.randn(
                        initial.shape, generator=generator,
                        device=self.device, dtype=initial.dtype,
                    )
                    theta = mu + scale * epsilon
                    log_posterior = self._joint_head_log_posterior(
                        theta, fold_representations, duration[train_idx],
                        event[train_idx], responsibilities[train_idx],
                    )
                    log_q = torch.distributions.Normal(mu, scale).log_prob(
                        theta
                    ).sum()
                    elbo_samples.append(log_posterior - log_q)
                loss = -torch.stack(elbo_samples).mean() / len(train_idx)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_((mu, rho), 10.0)
                optimizer.step()
                final_loss = float(loss.detach())

            with torch.no_grad():
                scale = F.softplus(rho).clamp_min(1e-5)
                heldout_representations = {
                    name: value[heldout_idx]
                    for name, value in representations.items()
                }
                draw_risks = {name: [] for name in self.expert_names}
                draw_loglik = []
                for _ in range(self.oof_vi_samples):
                    theta = mu + scale * torch.randn(
                        initial.shape, generator=generator,
                        device=self.device, dtype=initial.dtype,
                    )
                    for expert_index, name in enumerate(self.expert_names):
                        beta = theta[expert_index * d:(expert_index + 1) * d]
                        draw_risks[name].append(
                            heldout_representations[name].mv(beta)
                        )
                    draw_loglik.append(self._log_likelihood_from_theta(
                        theta, heldout_representations, duration[heldout_idx],
                        event[heldout_idx],
                    ))
                for name in self.expert_names:
                    values = torch.stack(draw_risks[name], dim=1)
                    oof_means[name][heldout_idx] = values.mean(1, keepdim=True)
                    oof_stds[name][heldout_idx] = values.std(
                        1, unbiased=False, keepdim=True
                    ).clamp_min(1e-6)
                values = torch.stack(draw_loglik, dim=0)
                oof_loglik[heldout_idx] = (
                    torch.logsumexp(values, dim=0)
                    - math.log(self.oof_vi_samples)
                )
            fold_records.append({
                "fold": fold + 1,
                "train_size": len(train_np),
                "heldout_size": len(heldout_np),
                "final_vi_loss": final_loss,
            })

        sorted_loglik = torch.sort(oof_loglik, dim=1).values
        self.oof_diagnostics_ = {
            "scope": "survival_head",
            "n_folds": n_folds,
            "mean_top_two_loglik_margin": float(
                (sorted_loglik[:, -1] - sorted_loglik[:, -2]).mean()
            ),
            "likelihood_winner_counts": torch.argmax(
                oof_loglik, dim=1
            ).bincount(minlength=3).cpu().tolist(),
            "folds": fold_records,
        }
        return oof_means, oof_stds, oof_loglik

    @staticmethod
    def _joint_posterior_soft_risk(gate_samples, risk_samples):
        """Compute E[sum_k pi_k(theta) r_k(theta)] over posterior draws."""
        gates = torch.stack(gate_samples)
        risks = torch.stack(risk_samples)
        if gates.shape != risks.shape:
            raise ValueError("Gate and expert-risk posterior draws must align")
        return torch.sum(gates * risks, dim=2).mean(0)

    def _refit_router(self, router_state, responsibility):
        router_optimizer = torch.optim.Adam(
            self.router.parameters(), lr=self.learning_rate
        )
        for _ in range(self.router_refit_epochs):
            gate = torch.softmax(self.router(router_state), dim=1)
            router_ce = -torch.mean(torch.sum(
                responsibility * torch.log(gate.clamp_min(1e-8)), dim=1
            ))
            router_kl = torch.mean(torch.sum(
                gate * (
                    torch.log(gate.clamp_min(1e-8))
                    - torch.log(self.reliability_prior).unsqueeze(0)
                ), dim=1
            ))
            router_loss = router_ce + self.beta_router_prior * router_kl
            router_optimizer.zero_grad()
            router_loss.backward()
            router_optimizer.step()

    def fit(self, x_rad, x_path, y, validation_fraction=0.2):
        if self.representation_normalized_:
            raise RuntimeError("Create a new model instance before refitting")
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
        best, best_loss, best_record, stale = None, math.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.train()
            optimizer.zero_grad()
            losses = []
            train_parts = []
            for _ in range(self.mc_train_samples):
                loss, parts_train = self._objective(
                    *[tensor[train_idx] for tensor in tensors], stochastic=True
                )
                losses.append(loss)
                train_parts.append(parts_train)
            train_loss = torch.stack(losses).mean()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss, parts = self._objective(
                    *[tensor[val_idx] for tensor in tensors], stochastic=False
                )
            value = val_loss.item()
            train_component_means = {
                name: torch.stack([part[name] for part in train_parts]).mean().item()
                for name in ("expert_nll", "router_ce", "router_kl")
            }
            record = {
                "epoch": epoch + 1, "train_loss": train_loss.item(),
                "val_loss": value, "expert_nll": parts["expert_nll"].item(),
                "router_ce": parts["router_ce"].item(),
                "router_kl": parts["router_kl"].item(),
                "weighted_router_kl": (
                    self.beta_router_prior * parts["router_kl"].item()
                ),
                "train_expert_nll": train_component_means["expert_nll"],
                "train_router_ce": train_component_means["router_ce"],
                "train_router_kl": train_component_means["router_kl"],
                "train_weighted_router_kl": (
                    self.beta_router_prior * train_component_means["router_kl"]
                ),
            }
            self.history_.append(record)
            if self.verbose and (
                epoch == 0 or (epoch + 1) % self.log_every == 0
            ):
                print(
                    f"  Epoch {epoch + 1}/{self.max_epochs}: "
                    f"Train={record['train_loss']:.4f}, Val={value:.4f}, "
                    f"Expert NLL={record['expert_nll']:.4f}, "
                    f"Router CE={record['router_ce']:.4f}, "
                    f"Router KL={record['router_kl']:.4f} "
                    f"(weighted={record['weighted_router_kl']:.4f})"
                )
            if value < best_loss - 1e-6:
                best_loss, stale = value, 0
                best = copy.deepcopy(self.state_dict())
                best_record = record.copy()
            else:
                stale += 1
                if stale >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
        if best is None:
            raise RuntimeError("Variational survival mixture produced no checkpoint")
        self.load_state_dict(best)
        self.best_epoch_ = best_record["epoch"]
        self.best_validation_terms_ = best_record
        # Empirical-Bayes stage: freeze learned representations, then sample
        # the conditional posterior of each compact linear survival head.
        self.eval()
        with torch.no_grad():
            _, representations = self._encode(tensors[0], tensors[1])
            representations = self._fit_representation_normalization(
                representations
            )
        # Start from the prespecified reliability prior. Patient-specific
        # assignments below are based only on cross-fitted expert evidence.
        posterior_responsibility = self.reliability_prior.unsqueeze(0).repeat(
            len(tensors[2]), 1
        ).detach()
        self.bayesian_em_history_ = []
        frozen_representations = {
            name: value.detach() for name, value in representations.items()
        }
        for em_iteration in range(self.bayesian_em_iterations):
            oof_means, oof_stds, oof_loglik = self._oof_variational_evidence(
                frozen_representations, tensors[2], tensors[3],
                posterior_responsibility,
            )
            with torch.no_grad():
                router_state = self._make_router_state(
                    oof_means, oof_stds
                ).detach()
                oof_gate = torch.softmax(self.router(router_state), dim=1)
                updated_responsibility = self._update_responsibilities(
                    oof_gate, oof_loglik
                )
                responsibility_change = torch.mean(torch.abs(
                    updated_responsibility - posterior_responsibility
                )).item()
            self._refit_router(router_state, updated_responsibility)
            posterior_responsibility = updated_responsibility

            # Fit the deployable full-data posterior using responsibilities
            # obtained strictly from held-out expert predictions.
            self.posterior_samples_, self.mcmc_diagnostics_ = {}, {}
            self._hmc_joint_head(
                frozen_representations, tensors[2], tensors[3],
                posterior_responsibility,
            )
            with torch.no_grad():
                for name in self.expert_names:
                    self.experts[name].risk_coef.copy_(
                        self.posterior_samples_[name]["beta"]
                        .mean(0).to(self.device)
                    )
                self.shared_log_baseline_hazard.copy_(
                    self.posterior_samples_["shared_log_baseline_hazard"]
                    .mean(0).to(self.device)
                )
            diagnostic = self.mcmc_diagnostics_["joint"]
            record = {
                "iteration": em_iteration + 1,
                "mean_absolute_responsibility_change": responsibility_change,
                "mean_responsibility_R": float(
                    posterior_responsibility[:, 0].mean()
                ),
                "mean_responsibility_P": float(
                    posterior_responsibility[:, 1].mean()
                ),
                "mean_responsibility_RP": float(
                    posterior_responsibility[:, 2].mean()
                ),
                **diagnostic,
            }
            self.bayesian_em_history_.append(record)
            if self.verbose:
                print(
                    f"  Bayesian EM {em_iteration + 1}: "
                    f"|delta r|={responsibility_change:.5f}, "
                    f"mean r R/P/RP="
                    f"{record['mean_responsibility_R']:.3f}/"
                    f"{record['mean_responsibility_P']:.3f}/"
                    f"{record['mean_responsibility_RP']:.3f}, "
                    f"mean accept prob="
                    f"{diagnostic['mean_accept_probability']:.3f} "
                    f"(target={diagnostic['target_accept_probability']:.2f}), "
                    f"state-change rate="
                    f"{diagnostic['state_change_rate']:.3f}, "
                    f"divergences={diagnostic['divergences']}, "
                    f"R-hat(max)={diagnostic['max_rhat']:.3f}, "
                    f"ESS(min)={diagnostic['min_ess']:.1f}"
                )
                print(
                    "    OOF evidence: winner counts R/P/RP="
                    + "/".join(str(value) for value in self.oof_diagnostics_[
                        "likelihood_winner_counts"
                    ])
                    + ", mean top-two log-likelihood margin="
                    f"{self.oof_diagnostics_['mean_top_two_loglik_margin']:.4f}"
                )
                for name, values in diagnostic["predictive_diagnostics"].items():
                    print(
                        f"    Predictive {name}: R-hat(max/median)="
                        f"{values['max_rhat']:.3f}/{values['median_rhat']:.3f}, "
                        f"p95={values['p95_rhat']:.3f}, "
                        f"ESS(min)={values['min_ess']:.1f}, "
                        f"worst patient={values['worst_patient_index']}"
                    )
                if "warning" in diagnostic:
                    print(f"    NUTS warning: {diagnostic['warning']}")
            if responsibility_change < self.responsibility_tolerance:
                break
        # Keep the two assignment distributions separate. Responsibilities use
        # observed outcomes and are available only during training; gate
        # probabilities depend on states alone and are the deployable policy.
        self.eval()
        with torch.no_grad():
            final_gate = torch.softmax(self.router(router_state), dim=1)
        self.training_responsibilities_ = (
            posterior_responsibility.cpu().numpy()
        )
        self.training_gate_probs_ = final_gate.cpu().numpy()
        self.training_gate_diagnostics_ = self.probability_diagnostics(
            self.training_gate_probs_
        )
        self.training_responsibility_diagnostics_ = self.probability_diagnostics(
            self.training_responsibilities_
        )
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
            _, representations = self._encode(x_rad, x_path)
            posterior_means, posterior_stds = self._posterior_log_risk_summary(
                representations
            )
            available = self.posterior_samples_["shared_log_baseline_hazard"].shape[0]
            draw_indices = torch.linspace(
                0, available - 1, steps=min(mc_samples, available)
            ).long()
            widths = self.time_boundaries_[1:] - self.time_boundaries_[:-1]
            exposure = torch.clamp(
                self.risk_horizon_ - self.time_boundaries_[:-1], min=0.0
            )
            exposure = torch.minimum(exposure, widths)
            for draw_index in draw_indices:
                draw_log_risks, draw_hazards = {}, []
                log_h0 = self.posterior_samples_[
                    "shared_log_baseline_hazard"
                ][draw_index].to(self.device).clamp(-12.0, 8.0)
                baseline = torch.sum(exposure * torch.exp(log_h0))
                for name in self.expert_names:
                    samples = self.posterior_samples_[name]
                    beta = samples["beta"][draw_index].to(self.device)
                    eta = representations[name].mv(beta).clamp(-12.0, 12.0)
                    draw_log_risks[name] = eta.unsqueeze(1)
                    draw_hazards.append(baseline * torch.exp(eta))
                gate = self._router_probs(draw_log_risks, posterior_stds)
                risks = torch.stack(draw_hazards, dim=1)
                gate_samples.append(gate)
                risk_samples.append(risks)
        probs = torch.stack(gate_samples).mean(0)
        expert_risks = torch.stack(risk_samples).mean(0)
        actions = torch.argmax(probs, dim=1)
        if hard:
            risk = expert_risks.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            risk = self._joint_posterior_soft_risk(
                gate_samples, risk_samples
            )
        uncertainty = torch.stack(gate_samples).var(0, unbiased=False).mean(1)
        # Posterior-coherent mixture E[sum_k pi_k(theta) r_k(theta)].
        # Multiplying separate posterior means would discard their covariance.
        soft_risk = self._joint_posterior_soft_risk(
            gate_samples, risk_samples
        )
        diagnostics = {
            "expert_risks": expert_risks.cpu().numpy(),
            "soft_risk": soft_risk.cpu().numpy(),
            "hard_risk": expert_risks.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1).cpu().numpy(),
            "policy": self.probability_diagnostics(probs.cpu().numpy()),
        }
        self.last_prediction_diagnostics_ = diagnostics
        return (
            risk.cpu().numpy(), actions.cpu().numpy(), probs.cpu().numpy(),
            uncertainty.cpu().numpy(), diagnostics,
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
        risk, actions, probs, uncertainty, diagnostics = self.model.predict(
            x_rad, x_path, hard=self.hard
        )
        self.actions_, self.probs_, self.uncertainty_ = actions, probs, uncertainty
        self.prediction_diagnostics_ = diagnostics
        return risk
