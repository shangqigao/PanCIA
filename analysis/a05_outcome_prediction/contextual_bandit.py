"""Legacy EM contextual-bandit survival model.

This module contains the original policy-loss, Torch Cox, contextual-bandit,
and pipeline implementations.  It is kept separate from the conditional
variational survival mixture so the two statistical approaches remain explicit.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PolicyNetwork(nn.Module):
    """
    Neural network policy that outputs probabilities for each action.
    
    Input: State vector [R, P, RP, |R-P|] on a shared robust OOF scale
    Output: Softmax probabilities for actions [Rad, Path, RP]
    """
    
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=3, dropout_rate=0.1):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=1)
    
    def get_logits(self, x):
        """Get raw logits for gradient computation."""
        return self.network(x)


class WeightedCoxPLLoss(nn.Module):
    """
    Negative Weighted Cox Partial Likelihood Loss with Exploration Mechanisms.
    
    This loss function combines:
    1. Weighted Cox PL (exploitation - maximize survival ranking)
    2. Entropy bonus (exploration - encourage action diversity)
    3. Uncertainty bonus (exploration - reward high-variance decisions)
    4. Temperature annealing (gradual shift from exploration to exploitation)
    """
    
    def __init__(self, 
                 entropy_weight=0.1,
                 uncertainty_weight=0.05,
                 min_entropy_weight=0.01,
                 max_entropy_weight=0.5,
                 temperature=1.0,
                 min_temperature=0.1,
                 annealing_rate=0.95):
        """
        Parameters
        ----------
        entropy_weight : float
            Initial weight for entropy bonus (higher = more exploration)
        uncertainty_weight : float
            Weight for uncertainty bonus (higher = explore uncertain actions)
        min_entropy_weight : float
            Minimum entropy weight (prevents complete exploitation)
        max_entropy_weight : float
            Maximum entropy weight (caps exploration)
        temperature : float
            Initial softmax temperature (higher = more uniform exploration)
        min_temperature : float
            Minimum temperature (sharpens policy over time)
        annealing_rate : float
            Rate at which temperature and entropy weight decrease
        """
        super(WeightedCoxPLLoss, self).__init__()
        
        self.entropy_weight = entropy_weight
        self.uncertainty_weight = uncertainty_weight
        self.min_entropy_weight = min_entropy_weight
        self.max_entropy_weight = max_entropy_weight
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.annealing_rate = annealing_rate
        
        # Store for adaptive adjustment
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def update_parameters(self, step_count=None):
        """
        Update temperature and entropy weight based on training progress.
        This implements exploration annealing.
        """
        if step_count is not None:
            self.step_count = step_count
        
        # Anneal temperature (gradually sharpen policy)
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.annealing_rate
        )
        
        # Anneal entropy weight (gradually reduce exploration)
        self.entropy_weight = max(
            self.min_entropy_weight,
            self.entropy_weight * self.annealing_rate
        )
        
        return self.temperature, self.entropy_weight
    
    def compute_entropy(self, probs):
        """
        Compute entropy of policy distribution.
        Higher entropy = more uniform action selection (exploration).
        """
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy.mean()
    
    def compute_uncertainty(self, probs, R, P, RP):
        """
        Compute uncertainty based on variance of risk scores across actions.
        Higher variance = policy is uncertain about which action is best.
        """
        # Weighted risk scores per action
        risk_all = torch.stack([R, P, RP], dim=1)  # (n_samples, 3)
        
        # Expected risk (already computed in weighted Cox PL)
        expected_risk = (probs * risk_all).sum(dim=1, keepdim=True)
        
        # Variance of risk across actions
        variance = torch.sum(probs * (risk_all - expected_risk)**2, dim=1)
        
        # Encourage exploration when variance is high (uncertainty bonus)
        return variance.mean()

    @staticmethod
    def compute_cox_loss(risk, E, T):
        """Breslow negative partial log-likelihood with tied-time support."""
        order = torch.argsort(T, descending=True, stable=True)
        risk_sorted = risk[order]
        time_sorted = T[order]
        event_sorted = E[order]

        _, group_ids, group_counts = torch.unique_consecutive(
            time_sorted, return_inverse=True, return_counts=True
        )
        group_end = torch.cumsum(group_counts, dim=0) - 1
        group_event_risk = torch.zeros(
            len(group_counts), device=risk.device, dtype=risk.dtype
        ).scatter_add_(0, group_ids, event_sorted * risk_sorted)
        group_event_count = torch.zeros_like(group_event_risk).scatter_add_(
            0, group_ids, event_sorted
        )
        group_log_risk = torch.logcumsumexp(risk_sorted, dim=0)[group_end]
        event_groups = group_event_count > 0
        log_likelihood = (
            group_event_risk[event_groups]
            - group_event_count[event_groups] * group_log_risk[event_groups]
        ).sum()
        return -log_likelihood / group_event_count[event_groups].sum().clamp_min(1.0)
    
    def forward(self, probs, R, P, RP, E, T,
                return_components=False, regularization_probs=None,
                exploration_prior=None):
        """
        Compute loss with exploration bonuses.
        
        Parameters
        ----------
        probs : torch.Tensor
            Policy probabilities for each action (n_samples, 3)
        R, P, RP : torch.Tensor
            Risk scores from each modality (n_samples,)
        E : torch.Tensor
            Event indicators (n_samples,)
        T : torch.Tensor
            Survival times (n_samples,)
        return_components : bool
            If True, return individual loss components for monitoring
            
        Returns
        -------
        loss : torch.Tensor
            Total loss (negative weighted Cox PL + exploration bonuses)
        """
        # ============================================================
        # COMPONENT 1: WEIGHTED COX PARTIAL LIKELIHOOD (Exploitation)
        # ============================================================
        
        reg_probs = probs if regularization_probs is None else regularization_probs

        # Compute weighted risk scores
        h_weighted = (probs[:, 0] * R + 
                      probs[:, 1] * P + 
                      probs[:, 2] * RP)
        
        # Apply temperature scaling for sharper/softer decisions
        # Higher temperature = softer (more exploration)
        # Lower temperature = sharper (more exploitation)
        h_weighted = h_weighted / self.temperature
        
        cox_loss = self.compute_cox_loss(h_weighted, E, T)
        
        # ============================================================
        # COMPONENT 2: ENTROPY BONUS (Exploration)
        # ============================================================
        
        # Compute entropy of policy distribution
        entropy = self.compute_entropy(reg_probs)
        
        # Entropy bonus: encourage exploration when entropy is low
        # We want to maximize entropy (minimize -entropy)
        entropy_bonus = -self.entropy_weight * entropy
        
        # ============================================================
        # COMPONENT 3: UNCERTAINTY BONUS (Exploration)
        # ============================================================
        
        # Encourage exploration when uncertainty is high
        uncertainty = self.compute_uncertainty(reg_probs, R, P, RP)
        uncertainty_bonus = self.uncertainty_weight * uncertainty
        
        # ============================================================
        # COMPONENT 4: ACTION DIVERSITY REGULARIZATION (Exploration)
        # ============================================================
        
        # Penalize if policy assigns near-zero probability to any action
        # This ensures all actions remain possible
        min_prob = torch.min(reg_probs, dim=1)[0]
        diversity_penalty = -torch.log(min_prob + 1e-8).mean()
        diversity_weight = 0.01  # Small weight to avoid over-regularization
        
        # ============================================================
        # COMBINE LOSS COMPONENTS
        # ============================================================
        
        total_loss = (cox_loss + 
                      entropy_bonus + 
                      uncertainty_bonus + 
                      diversity_weight * diversity_penalty)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy,
                'uncertainty_bonus': uncertainty_bonus,
                'uncertainty_value': uncertainty,
                'diversity_penalty': diversity_penalty,
                'temperature': self.temperature,
                'entropy_weight': self.entropy_weight
            }
        
        return total_loss


class BayesianWeightedCoxPLLoss(nn.Module):
    """
    Bayesian-inspired loss with Thompson sampling style exploration.
    
    This loss adds noise to the risk scores based on policy uncertainty,
    encouraging the policy to explore actions with high epistemic uncertainty.
    """
    
    def __init__(self, 
                 noise_scale=0.1,
                 min_noise_scale=0.01,
                 exploration_bonus_weight=0.1,
                 annealing_rate=0.95):
        """
        Parameters
        ----------
        noise_scale : float
            Initial noise scale for risk scores
        min_noise_scale : float
            Minimum noise scale (prevents complete exploitation)
        exploration_bonus_weight : float
            Weight for exploration bonus based on uncertainty
        annealing_rate : float
            Rate at which noise scale decreases
        """
        super(BayesianWeightedCoxPLLoss, self).__init__()
        
        self.noise_scale = noise_scale
        self.min_noise_scale = min_noise_scale
        self.exploration_bonus_weight = exploration_bonus_weight
        self.annealing_rate = annealing_rate
        
        self.step_count = 0
        
    def update_parameters(self, step_count=None):
        """Annealing: gradually reduce exploration noise."""
        if step_count is not None:
            self.step_count = step_count
        
        self.noise_scale = max(
            self.min_noise_scale,
            self.noise_scale * self.annealing_rate
        )
        
        return self.noise_scale
    
    def compute_epistemic_uncertainty(self, probs, R, P, RP):
        """
        Compute epistemic uncertainty using dropout-like variance.
        Higher uncertainty = encourage more exploration.
        """
        # Standard deviation of risk scores across actions
        risk_all = torch.stack([R, P, RP], dim=1)
        std_risk = torch.std(risk_all, dim=1)
        return std_risk.mean()
    
    def forward(self, probs, R, P, RP, E, T,
                return_components=False, regularization_probs=None,
                exploration_prior=None):
        """
        Forward pass with Bayesian exploration.
        """
        reg_probs = probs if regularization_probs is None else regularization_probs

        # ============================================================
        # COMPONENT 1: BAYESIAN RISK SCORES (Exploration via noise)
        # ============================================================
        
        # Add noise to risk scores proportional to uncertainty
        # This implements a simple form of Thompson sampling
        noise = torch.randn_like(R) * self.noise_scale
        R_noisy = R + noise * (probs[:, 0] * 0.1 + 0.1)  # More noise when action prob is low
        
        noise = torch.randn_like(P) * self.noise_scale
        P_noisy = P + noise * (probs[:, 1] * 0.1 + 0.1)
        
        noise = torch.randn_like(RP) * self.noise_scale
        RP_noisy = RP + noise * (probs[:, 2] * 0.1 + 0.1)
        
        # Weighted risk with noisy scores
        h_weighted = (probs[:, 0] * R_noisy + 
                      probs[:, 1] * P_noisy + 
                      probs[:, 2] * RP_noisy)
        
        cox_loss = WeightedCoxPLLoss.compute_cox_loss(h_weighted, E, T)
        
        # ============================================================
        # COMPONENT 2: EXPLORATION BONUS (epistemic uncertainty)
        # ============================================================
        
        # Encourage exploration when epistemic uncertainty is high
        epistemic_uncertainty = self.compute_epistemic_uncertainty(probs, R, P, RP)
        exploration_bonus = -self.exploration_bonus_weight * epistemic_uncertainty
        
        # ============================================================
        # COMPONENT 3: STANDARD ENTROPY BONUS
        # ============================================================
        
        entropy = -torch.sum(
            reg_probs * torch.log(reg_probs + 1e-8), dim=1
        ).mean()
        entropy_bonus = -0.05 * entropy
        
        # ============================================================
        # COMBINE
        # ============================================================
        
        total_loss = cox_loss + exploration_bonus + entropy_bonus
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'exploration_bonus': exploration_bonus,
                'epistemic_uncertainty': epistemic_uncertainty,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy,
                'noise_scale': self.noise_scale
            }
        
        return total_loss


class AdaptiveWeightedCoxPLLoss(nn.Module):
    """Historical adaptive entropy/diversity/disagreement exploration."""
    
    def __init__(self,
                 initial_exploration_weight=0.3,
                 min_exploration_weight=0.01,
                 max_exploration_weight=0.5,
                 plateau_threshold=0.001,
                 plateau_patience=5):
        """
        Parameters
        ----------
        initial_exploration_weight : float
            Initial weight for exploration bonuses
        min_exploration_weight : float
            Minimum exploration weight
        max_exploration_weight : float
            Maximum exploration weight
        plateau_threshold : float
            Minimum loss improvement to detect plateau
        plateau_patience : int
            Number of steps before considering plateau
        """
        super(AdaptiveWeightedCoxPLLoss, self).__init__()
        
        self.exploration_weight = initial_exploration_weight
        self.min_exploration_weight = min_exploration_weight
        self.max_exploration_weight = max_exploration_weight
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        
        # Tracking for adaptive adjustment
        self.loss_history = []
        self.plateau_counter = 0
        self.step_count = 0
        self.best_loss = float('inf')
        
    def update_exploration_weight(self, loss_value):
        """Increase exploration on plateaus and decay it otherwise."""
        self.step_count += 1
        value = loss_value.item() if torch.is_tensor(loss_value) else loss_value
        self.loss_history.append(value)
        if len(self.loss_history) > 1:
            improvement = self.loss_history[-2] - self.loss_history[-1]
            if abs(improvement) < self.plateau_threshold:
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
            if self.plateau_counter >= self.plateau_patience:
                self.exploration_weight = min(
                    self.max_exploration_weight,
                    self.exploration_weight * 1.2,
                )
                self.plateau_counter = 0
            else:
                self.exploration_weight = max(
                    self.min_exploration_weight,
                    self.exploration_weight * 0.99,
                )
        return self.exploration_weight

    @staticmethod
    def compute_action_diversity(probs):
        avg_probs = probs.mean(dim=0)
        uniform = torch.ones_like(avg_probs) / avg_probs.shape[0]
        return torch.sum(
            avg_probs * (
                torch.log(avg_probs + 1e-8) - torch.log(uniform + 1e-8)
            )
        )
    
    def forward(self, probs, R, P, RP, E, T, return_components=False,
                regularization_probs=None, exploration_prior=None):
        """
        Forward pass with adaptive exploration.
        """
        reg_probs = probs if regularization_probs is None else regularization_probs

        # ============================================================
        # COMPONENT 1: WEIGHTED COX PL (Exploitation)
        # ============================================================
        
        h_weighted = (probs[:, 0] * R + 
                      probs[:, 1] * P + 
                      probs[:, 2] * RP)
        
        cox_loss = WeightedCoxPLLoss.compute_cox_loss(h_weighted, E, T)
        
        entropy = -torch.sum(
            reg_probs * torch.log(reg_probs + 1e-8), dim=1
        ).mean()
        entropy_bonus = -self.exploration_weight * entropy
        diversity = self.compute_action_diversity(reg_probs)
        diversity_bonus = self.exploration_weight * 0.1 * diversity

        risk_all = torch.stack([R, P, RP], dim=1)
        expected_risk = (reg_probs * risk_all).sum(dim=1, keepdim=True)
        variance = torch.sum(
            reg_probs * (risk_all - expected_risk).square(), dim=1
        )
        uncertainty_bonus = (
            self.exploration_weight * 0.1 * variance.mean()
        )
        
        # ============================================================
        # COMBINE
        # ============================================================
        
        total_loss = (
            cox_loss + entropy_bonus + diversity_bonus + uncertainty_bonus
        )
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy,
                'diversity_bonus': diversity_bonus,
                'diversity_value': diversity,
                'uncertainty_bonus': uncertainty_bonus,
                'uncertainty_value': variance.mean(),
                'exploration_weight': self.exploration_weight
            }
        
        return total_loss


class EnsembleWeightedCoxPLLoss(nn.Module):
    """
    Ensemble-based loss with exploration via multiple hypotheses.
    
    Maintains multiple risk estimates and encourages the policy to
    explore actions that are favored by different ensemble members.
    """
    
    def __init__(self, 
                 ensemble_size=5,
                 exploration_weight=0.1,
                 ensemble_noise_scale=0.05):
        """
        Parameters
        ----------
        ensemble_size : int
            Number of ensemble members (different risk estimates)
        exploration_weight : float
            Weight for exploration bonus
        ensemble_noise_scale : float
            Noise scale for ensemble diversity
        """
        super(EnsembleWeightedCoxPLLoss, self).__init__()
        
        self.ensemble_size = ensemble_size
        self.exploration_weight = exploration_weight
        self.ensemble_noise_scale = ensemble_noise_scale
        
    def forward(self, probs, R, P, RP, E, T,
                return_components=False, regularization_probs=None,
                exploration_prior=None):
        """
        Forward pass with ensemble exploration.
        """
        reg_probs = probs if regularization_probs is None else regularization_probs

        # ============================================================
        # COMPONENT 1: ENSEMBLE RISK ESTIMATES
        # ============================================================
        
        # Create ensemble of perturbed risk scores
        ensemble_losses = []
        
        for e in range(self.ensemble_size):
            # Add noise to risk scores (different for each ensemble member)
            noise_scale = self.ensemble_noise_scale * (1 + 0.1 * e / self.ensemble_size)
            
            R_e = R + torch.randn_like(R) * noise_scale
            P_e = P + torch.randn_like(P) * noise_scale
            RP_e = RP + torch.randn_like(RP) * noise_scale
            
            # Compute weighted risk
            h_e = (probs[:, 0] * R_e + 
                   probs[:, 1] * P_e + 
                   probs[:, 2] * RP_e)
            ensemble_losses.append(
                WeightedCoxPLLoss.compute_cox_loss(h_e, E, T)
            )
        
        # Average over ensemble
        cox_loss = torch.stack(ensemble_losses).mean()
        
        # ============================================================
        # COMPONENT 2: ENSEMBLE UNCERTAINTY (Exploration)
        # ============================================================
        
        # Variance across ensemble members
        ensemble_losses_tensor = torch.stack(ensemble_losses)
        ensemble_variance = torch.var(ensemble_losses_tensor)
        
        # Encourage exploration when ensemble members disagree
        exploration_bonus = -self.exploration_weight * ensemble_variance
        
        # ============================================================
        # COMPONENT 3: STANDARD ENTROPY BONUS
        # ============================================================
        
        entropy = -torch.sum(
            reg_probs * torch.log(reg_probs + 1e-8), dim=1
        ).mean()
        entropy_bonus = -0.05 * entropy
        
        # ============================================================
        # COMBINE
        # ============================================================
        
        total_loss = cox_loss + exploration_bonus + entropy_bonus
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'exploration_bonus': exploration_bonus,
                'ensemble_variance': ensemble_variance,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy
            }
        
        return total_loss


class TorchCoxPH:
    """Linear Cox proportional-hazards model optimized with PyTorch.

    The implementation supports case weights, Breslow handling of tied event
    times, elastic-net regularization, warm starts, and CUDA execution.  It is
    intentionally prediction-focused: unlike ``lifelines.CoxPHFitter``, it does
    not calculate standard errors or a robust covariance matrix.
    """

    def __init__(self, penalizer=0.0, l1_ratio=0.0, learning_rate=0.05,
                 max_epochs=500, tolerance=1e-6, patience=20,
                 gradient_clip=10.0, device='cpu'):
        if penalizer < 0:
            raise ValueError("penalizer must be non-negative")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between 0 and 1")

        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.gradient_clip = float(gradient_clip)
        self.device = torch.device(
            device if str(device).startswith('cuda') and torch.cuda.is_available()
            else 'cpu'
        )
        self.coef_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.loss_ = None

    @staticmethod
    def _as_tensor(values, device, dtype=torch.float32):
        if torch.is_tensor(values):
            return values.detach().to(device=device, dtype=dtype)
        return torch.as_tensor(values, device=device, dtype=dtype)

    @staticmethod
    def _validate_inputs(X, T, E, weights):
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        n_samples = X.shape[0]
        if T.ndim != 1 or E.ndim != 1 or weights.ndim != 1:
            raise ValueError("T, E, and weights must be one-dimensional")
        if not (len(T) == len(E) == len(weights) == n_samples):
            raise ValueError("X, T, E, and weights must contain the same samples")
        if n_samples < 2:
            raise ValueError("At least two samples are required")
        if not torch.isfinite(X).all() or not torch.isfinite(T).all():
            raise ValueError("X and T must contain only finite values")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if torch.sum(weights * E) <= 0:
            raise ValueError("At least one event must have a positive weight")

    @staticmethod
    def _prepare_risk_sets(X, T, E, weights):
        order = torch.argsort(T, descending=True, stable=True)
        X_sorted = X[order]
        T_sorted = T[order]
        E_sorted = E[order]
        W_sorted = weights[order]

        _, group_ids, group_counts = torch.unique_consecutive(
            T_sorted, return_inverse=True, return_counts=True
        )
        group_end = torch.cumsum(group_counts, dim=0) - 1
        return X_sorted, E_sorted, W_sorted, group_ids, group_end

    @staticmethod
    def _breslow_loss_from_risk_sets(beta, risk_sets):
        """Return mean weighted negative partial log-likelihood."""
        X_sorted, E_sorted, W_sorted, group_ids, group_end = risk_sets

        eta = X_sorted.mv(beta)
        log_weights = torch.where(
            W_sorted > 0,
            torch.log(W_sorted),
            torch.full_like(W_sorted, -torch.inf),
        )
        log_risk_sums = torch.logcumsumexp(eta + log_weights, dim=0)

        n_groups = group_end.numel()
        event_weights = W_sorted * E_sorted

        weighted_event_eta = torch.zeros(
            n_groups, device=X_sorted.device, dtype=X_sorted.dtype
        ).scatter_add_(0, group_ids, event_weights * eta)
        event_weight_sum = torch.zeros(
            n_groups, device=X_sorted.device, dtype=X_sorted.dtype
        ).scatter_add_(0, group_ids, event_weights)

        group_log_risk = log_risk_sums[group_end]
        # Groups with no weighted events contribute exactly zero. Mask them
        # before multiplication so an empty early risk set cannot form 0 * -inf.
        event_groups = event_weight_sum > 0
        log_likelihood = (
            weighted_event_eta[event_groups]
            - event_weight_sum[event_groups] * group_log_risk[event_groups]
        )
        total_event_weight = event_weight_sum[event_groups].sum()
        return -log_likelihood.sum() / total_event_weight

    @classmethod
    def _breslow_negative_log_likelihood(cls, beta, X, T, E, weights):
        """Convenience entry point that prepares and evaluates risk sets."""
        risk_sets = cls._prepare_risk_sets(X, T, E, weights)
        return cls._breslow_loss_from_risk_sets(beta, risk_sets)

    def _objective(self, beta, risk_sets, include_l1=True):
        loss = self._breslow_loss_from_risk_sets(beta, risk_sets)
        l2_strength = self.penalizer * (1.0 - self.l1_ratio)
        if l2_strength:
            loss = loss + 0.5 * l2_strength * torch.sum(beta.square())
        if include_l1:
            l1_strength = self.penalizer * self.l1_ratio
            if l1_strength:
                loss = loss + l1_strength * torch.sum(torch.abs(beta))
        return loss

    def fit(self, X, T, E, weights=None, initial_coef=None):
        X_tensor = self._as_tensor(X, self.device)
        T_tensor = self._as_tensor(T, self.device).reshape(-1)
        E_tensor = self._as_tensor(E, self.device).reshape(-1)
        if weights is None:
            W_tensor = torch.ones_like(T_tensor)
        else:
            W_tensor = self._as_tensor(weights, self.device).reshape(-1)

        self._validate_inputs(X_tensor, T_tensor, E_tensor, W_tensor)
        self.n_features_in_ = X_tensor.shape[1]
        risk_sets = self._prepare_risk_sets(
            X_tensor, T_tensor, E_tensor, W_tensor
        )

        if initial_coef is None:
            beta = torch.zeros(
                self.n_features_in_, device=self.device, dtype=X_tensor.dtype
            )
        else:
            beta = self._as_tensor(initial_coef, self.device).reshape(-1).clone()
            if beta.numel() != self.n_features_in_:
                raise ValueError("initial_coef has the wrong number of features")
        beta.requires_grad_(True)

        optimizer = optim.Adam([beta], lr=self.learning_rate)
        best_coef = beta.detach().clone()
        best_loss = float('inf')
        previous_loss = None
        stale_epochs = 0
        l1_strength = self.penalizer * self.l1_ratio

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            smooth_loss = self._objective(
                beta, risk_sets, include_l1=False
            )
            if not torch.isfinite(smooth_loss):
                raise FloatingPointError("Non-finite TorchCoxPH loss")
            smooth_loss.backward()
            torch.nn.utils.clip_grad_norm_([beta], self.gradient_clip)
            optimizer.step()

            # Proximal step for the non-smooth L1 component.
            if l1_strength:
                threshold = self.learning_rate * l1_strength
                with torch.no_grad():
                    beta.copy_(
                        torch.sign(beta) * torch.clamp(torch.abs(beta) - threshold, min=0)
                    )

            with torch.no_grad():
                current_loss = self._objective(
                    beta, risk_sets
                ).item()
            if not np.isfinite(current_loss):
                raise FloatingPointError("Non-finite TorchCoxPH objective")

            if current_loss < best_loss:
                best_loss = current_loss
                best_coef = beta.detach().clone()

            if previous_loss is not None:
                relative_change = abs(previous_loss - current_loss) / max(
                    1.0, abs(previous_loss)
                )
                stale_epochs = stale_epochs + 1 if relative_change < self.tolerance else 0
                if stale_epochs >= self.patience:
                    self.n_iter_ = epoch + 1
                    break
            previous_loss = current_loss
            self.n_iter_ = epoch + 1

        self.coef_ = best_coef.detach().cpu().numpy().copy()
        self.loss_ = best_loss
        return self

    def predict_log_partial_hazard(self, X):
        if self.coef_ is None:
            raise RuntimeError("TorchCoxPH must be fitted before prediction")
        X_tensor = self._as_tensor(X, self.device)
        if X_tensor.ndim != 2 or X_tensor.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible feature dimension")
        coef = self._as_tensor(self.coef_, self.device)
        with torch.no_grad():
            return X_tensor.mv(coef).cpu().numpy()


class ContextualBandit:
    """
    Contextual Bandit with PyTorch policy network.
    
    Implements the EM framework with direct policy optimization:
    - E-Step: Update policy network by minimizing negative weighted Cox PL
    - M-Step: Update subgroup-specific Cox models with sample weights
    
    Parameters
    ----------
    alpha_range : list, default=[0.001, 0.01, 0.1, 1.0, 10.0]
        Elastic-net penalty range for Cox models
    max_iterations : int, default=10
        Maximum number of EM iterations
    convergence_threshold : float, default=0.001
        Minimum improvement in C-index for convergence
    hidden_dim : int, default=16
        Hidden layer dimension for policy network
    learning_rate : float, default=0.01
        Learning rate for policy network
    batch_size : int, default=32
        Retained for compatibility; policy Cox training uses full risk sets
    policy_epochs : int, default=50
        Number of epochs for policy training per EM iteration
    cv_folds : int, default=5
        Number of cross-validation folds for Cox model selection
    cox_learning_rate : float, default=0.05
        Learning rate for TorchCoxPH optimization
    cox_max_epochs : int, default=500
        Maximum optimizer steps for each TorchCoxPH fit
    cox_tolerance : float, default=1e-6
        Relative objective-change threshold for Cox early stopping
    cox_patience : int, default=20
        Consecutive low-change steps before stopping Cox optimization
    cox_l1_ratio : float, default=0.9
        Elastic-net mixing parameter used by TorchCoxPH
    policy_risk_clip : float, default=5.0
        Absolute clipping bound after OOF centering and common-scale conversion.
    rp_cost_weight : float, default=1.0
        Strength of the evidence-based penalty on RP policy probability
    rp_minimum_gain : float, default=0.01
        Required lower-confidence-bound C-index gain for cost-free RP use
    rp_bootstrap_samples : int, default=500
        Paired bootstrap samples used to estimate RP performance evidence
    hard_policy : bool, default=False
        Train Cox risk with straight-through one-hot Gumbel-Softmax actions
    gumbel_temperature : float, default=1.0
        Initial Gumbel-Softmax temperature
    device : str, default='cuda'
        Device for PyTorch ('cuda' or 'cpu')
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, 
                 alpha_range=None,
                 max_iterations=10,
                 convergence_threshold=0.001,
                 hidden_dim=16,
                 learning_rate=0.01,
                 batch_size=32,
                 policy_epochs=50,
                 cv_folds=5,
                 cox_learning_rate=0.05,
                 cox_max_epochs=500,
                 cox_tolerance=1e-6,
                 cox_patience=20,
                 cox_l1_ratio=0.9,
                 cox_gradient_clip=10.0,
                 min_expert_weight=0.01,
                 policy_risk_clip=5.0,
                 rp_cost_weight=1.0,
                 rp_minimum_gain=0.01,
                 rp_bootstrap_samples=500,
                 rp_confidence=0.95,
                 hard_policy=False,
                 gumbel_temperature=1.0,
                 gumbel_min_temperature=0.1,
                 gumbel_anneal_rate=0.95,
                 loss_type='adaptive',  # 'weighted', 'bayesian', 'adaptive', 'ensemble'
                 exploration_weight=0.1,
                 entropy_weight=0.05,
                 uncertainty_weight=0.05,
                 temperature=1.0,
                 device='cuda',
                 random_state=None):
        
        self.alpha_range = (
            [0.001, 0.01, 0.1, 1.0, 10.0]
            if alpha_range is None else list(alpha_range)
        )
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.cv_folds = cv_folds
        self.cox_learning_rate = cox_learning_rate
        self.cox_max_epochs = cox_max_epochs
        self.cox_tolerance = cox_tolerance
        self.cox_patience = cox_patience
        self.cox_l1_ratio = cox_l1_ratio
        self.cox_gradient_clip = cox_gradient_clip
        if not 0.0 <= min_expert_weight < 1.0 / 3.0:
            raise ValueError("min_expert_weight must be in [0, 1/3)")
        self.min_expert_weight = float(min_expert_weight)
        if policy_risk_clip <= 0:
            raise ValueError("policy_risk_clip must be positive")
        self.policy_risk_clip = float(policy_risk_clip)
        if rp_cost_weight < 0:
            raise ValueError("rp_cost_weight must be non-negative")
        if rp_bootstrap_samples < 0:
            raise ValueError("rp_bootstrap_samples must be non-negative")
        if not 0.0 < rp_confidence < 1.0:
            raise ValueError("rp_confidence must be between 0 and 1")
        self.rp_cost_weight = rp_cost_weight
        self.rp_minimum_gain = rp_minimum_gain
        self.rp_bootstrap_samples = rp_bootstrap_samples
        self.rp_confidence = rp_confidence
        if gumbel_temperature <= 0 or gumbel_min_temperature <= 0:
            raise ValueError("Gumbel temperatures must be positive")
        if not 0.0 < gumbel_anneal_rate <= 1.0:
            raise ValueError("gumbel_anneal_rate must be in (0, 1]")
        self.hard_policy = hard_policy
        self.gumbel_initial_temperature = gumbel_temperature
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_min_temperature = gumbel_min_temperature
        self.gumbel_anneal_rate = gumbel_anneal_rate
        self.loss_type = loss_type
        self.exploration_weight = exploration_weight
        self.entropy_weight = entropy_weight
        self.uncertainty_weight = uncertainty_weight
        self.temperature = temperature
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.random_state = random_state
        self._cox_warm_starts = {}
        
    def _init_loss_function(self):
        """Initialize the appropriate loss function."""
        if self.loss_type == 'weighted':
            # Original weighted Cox PL with entropy bonus
            self.policy_loss_fn = WeightedCoxPLLoss(
                entropy_weight=self.entropy_weight,
                uncertainty_weight=self.uncertainty_weight,
                temperature=self.temperature
            )
        elif self.loss_type == 'bayesian':
            # Bayesian exploration with noise
            self.policy_loss_fn = BayesianWeightedCoxPLLoss(
                noise_scale=0.1,
                exploration_bonus_weight=self.exploration_weight
            )
        elif self.loss_type == 'ensemble':
            # Ensemble-based exploration
            self.policy_loss_fn = EnsembleWeightedCoxPLLoss(
                ensemble_size=5,
                exploration_weight=self.exploration_weight
            )
        else:  # 'adaptive' (default)
            # Fully adaptive exploration-exploitation balancing
            self.policy_loss_fn = AdaptiveWeightedCoxPLLoss(
                initial_exploration_weight=self.exploration_weight,
                min_exploration_weight=0.01,
                max_exploration_weight=0.5,
            )
    
    def _fit_policy_risk_reference(self, risks, fit_indices, E, T,
                                   reliability_risks=None):
        """Fit separate full-risk z-scores and an OOF reliability prior."""
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        E = np.asarray(E, dtype=bool).reshape(-1)
        T = np.asarray(T, dtype=np.float32).reshape(-1)
        names = ('R', 'P', 'RP')
        centers = {}
        scales = {}
        cindices = {}

        if reliability_risks is None:
            reliability_risks = risks

        for name in names:
            values = np.asarray(risks[name], dtype=np.float32).reshape(-1)
            if len(values) != len(T):
                raise ValueError("OOF risks and survival outcomes must align")
            fit_values = values[fit_indices]
            if fit_values.size < 2 or not np.isfinite(fit_values).all():
                raise ValueError("Policy risk reference requires finite risks")
            centers[name] = float(np.mean(fit_values))
            scales[name] = max(float(np.std(fit_values)), 1e-6)
            reliability_values = np.asarray(
                reliability_risks[name], dtype=np.float32
            ).reshape(-1)
            cindices[name] = float(self._risk_cindex(
                reliability_values[fit_indices],
                E[fit_indices], T[fit_indices]
            ))

        return {
            'centers': centers,
            'scales': scales,
            'cindex': cindices,
        }

    @staticmethod
    def _apply_policy_risk_reference(risk, expert_name, reference):
        """Apply stored training-OOF risk statistics to any prediction batch."""
        risk = np.asarray(risk, dtype=np.float32)
        if not np.isfinite(risk).all():
            raise ValueError("Risk values must be finite")
        return (
            (risk - reference['centers'][expert_name])
            / reference['scales'][expert_name]
        ).astype(np.float32)

    def _normalize_expert_risks(self, R, P, RP):
        return (
            self._apply_policy_risk_reference(
                R, 'R', self.policy_risk_reference
            ),
            self._apply_policy_risk_reference(
                P, 'P', self.policy_risk_reference
            ),
            self._apply_policy_risk_reference(
                RP, 'RP', self.policy_risk_reference
            ),
        )

    @staticmethod
    def _make_policy_state(R, P, RP):
        """Build the original compact policy state."""
        return np.column_stack([R, P, np.abs(R - P)]).astype(np.float32)

    @staticmethod
    def _risk_cindex(risk, E, T):
        return concordance_index(T, -risk, E.astype(bool))

    def _prepare_expert_fit_weights(self, policy_weights, events):
        """Pass raw soft responsibilities to the normalized Cox objective."""
        policy_weights = np.asarray(policy_weights, dtype=np.float64).reshape(-1)
        events = np.asarray(events, dtype=bool).reshape(-1)
        if policy_weights.shape != events.shape:
            raise ValueError("policy_weights and events must have equal length")
        if not np.isfinite(policy_weights).all() or np.any(policy_weights < 0):
            raise ValueError("policy_weights must be finite and non-negative")

        event_weights = policy_weights[events]
        squared_sum = np.square(event_weights).sum()
        event_ess = (
            float(event_weights.sum() ** 2 / squared_sum)
            if squared_sum > 0 else 0.0
        )
        # Keep the EM output unchanged here. TorchCoxPH fixes the otherwise
        # arbitrary global case-weight scale inside its likelihood; patient-to-
        # patient responsibility ratios remain exactly those of the policy.
        return np.ascontiguousarray(policy_weights, dtype=np.float32), {
            'event_ess': event_ess,
            'min_weight': float(policy_weights.min()),
            'max_weight': float(policy_weights.max()),
        }

    def _compute_rp_cost(self, R, P, RP, E, T, bootstrap_indices=None,
                         seed_offset=0):
        """Estimate whether RP reliably improves on both unimodal experts.

        The same patient indices are used for all three experts in every
        bootstrap replicate. RP is cost-free only when the lower confidence
        bounds of both paired C-index gains exceed ``rp_minimum_gain``.
        """
        R = np.asarray(R, dtype=np.float32)
        P = np.asarray(P, dtype=np.float32)
        RP = np.asarray(RP, dtype=np.float32)
        E = np.asarray(E, dtype=bool)
        T = np.asarray(T, dtype=np.float32)

        point_scores = {
            'rad': self._risk_cindex(R, E, T),
            'path': self._risk_cindex(P, E, T),
            'rp': self._risk_cindex(RP, E, T),
        }
        point_gains = np.array([
            point_scores['rp'] - point_scores['rad'],
            point_scores['rp'] - point_scores['path'],
        ])

        gains = []
        if bootstrap_indices is not None:
            bootstrap_indices = np.asarray(bootstrap_indices, dtype=np.int64)
        elif self.rp_bootstrap_samples > 0:
            base_seed = 0 if self.random_state is None else self.random_state
            rng = np.random.default_rng(base_seed + seed_offset)
            n_samples = len(T)
            bootstrap_indices = rng.integers(
                0, n_samples, size=(self.rp_bootstrap_samples, n_samples)
            )

        if bootstrap_indices is not None:
            for idx in bootstrap_indices:
                try:
                    c_rad = self._risk_cindex(R[idx], E[idx], T[idx])
                    c_path = self._risk_cindex(P[idx], E[idx], T[idx])
                    c_rp = self._risk_cindex(RP[idx], E[idx], T[idx])
                    gains.append([c_rp - c_rad, c_rp - c_path])
                except Exception:
                    # Some small/censored resamples contain no comparable pair.
                    continue

        if gains:
            lower_percentile = 100.0 * (1.0 - self.rp_confidence) / 2.0
            lower_gains = np.percentile(np.asarray(gains), lower_percentile, axis=0)
        else:
            lower_gains = point_gains

        rp_evidence = float(np.min(lower_gains))
        rp_cost = max(0.0, self.rp_minimum_gain - rp_evidence)
        return rp_cost, {
            'cindex_rad': point_scores['rad'],
            'cindex_path': point_scores['path'],
            'cindex_rp': point_scores['rp'],
            'lower_gain_vs_rad': float(lower_gains[0]),
            'lower_gain_vs_path': float(lower_gains[1]),
            'valid_bootstraps': len(gains),
        }
    
    def train_survival_model(self, X, T, E, weights=None, alpha_range=None,
                             model_key='cox', alpha_selection_indices=None):
        """
        Train a CoxPH model with cross-validation for regularization parameter selection.
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        T : ndarray
            Survival times
        E : ndarray
            Event indicators (1=event, 0=censored)
        weights : ndarray, optional
            Sample weights
        alpha_range : list, optional
            Regularization parameter range
            
        Returns
        -------
        model : TorchCoxPH
            Fitted GPU-native CoxPH model with the best regularization parameter
        best_alpha : float
            Best regularization parameter
        """
        if alpha_range is None:
            alpha_range = self.alpha_range

        X = np.ascontiguousarray(X, dtype=np.float32)
        T = np.ascontiguousarray(T, dtype=np.float32).reshape(-1)
        E = np.ascontiguousarray(E, dtype=np.float32).reshape(-1)
        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
        
        # Cross-validation to select best alpha
        best_alpha = None
        best_concordance = -1
        best_oof_risk = None
        n_samples = len(T)
        indices = np.arange(n_samples)
        if alpha_selection_indices is None:
            alpha_selection_indices = indices
        alpha_selection_indices = np.asarray(
            alpha_selection_indices, dtype=np.int64
        )

        def make_model(alpha):
            return TorchCoxPH(
                penalizer=alpha,
                l1_ratio=self.cox_l1_ratio,
                learning_rate=self.cox_learning_rate,
                max_epochs=self.cox_max_epochs,
                tolerance=self.cox_tolerance,
                patience=self.cox_patience,
                gradient_clip=self.cox_gradient_clip,
                device=self.device,
            )
        
        for alpha in alpha_range:
            try:
                oof_risk = np.full(n_samples, np.nan, dtype=np.float32)
                
                # Preserve the existing contiguous-fold CV construction.
                for fold in range(self.cv_folds):
                    # Split indices
                    fold_size = n_samples // self.cv_folds
                    start = fold * fold_size
                    end = start + fold_size if fold < self.cv_folds - 1 else n_samples
                    
                    val_idx = indices[start:end]
                    train_idx = np.concatenate([indices[:start], indices[end:]])

                    if len(val_idx) == 0 or len(train_idx) < 2:
                        continue

                    warm_key = (model_key, float(alpha), 'fold', fold)
                    model = make_model(alpha)
                    fold_weights = None if weights is None else weights[train_idx]
                    model.fit(
                        X[train_idx], T[train_idx], E[train_idx],
                        weights=fold_weights,
                        initial_coef=self._cox_warm_starts.get(warm_key),
                    )
                    self._cox_warm_starts[warm_key] = model.coef_.copy()
                    
                    # Validate
                    try:
                        risk_scores = model.predict_log_partial_hazard(X[val_idx])
                        oof_risk[val_idx] = risk_scores
                    except Exception:
                        continue

                score_idx = alpha_selection_indices[
                    np.isfinite(oof_risk[alpha_selection_indices])
                ]
                if len(score_idx) < 2:
                    continue
                mean_cv_score = concordance_index(
                    T[score_idx], -oof_risk[score_idx],
                    E[score_idx].astype(bool)
                )
                
                if mean_cv_score > best_concordance:
                    best_concordance = mean_cv_score
                    best_alpha = alpha
                    best_oof_risk = oof_risk.copy()
                    
            except Exception as e:
                print(f"  Alpha={alpha} failed: {e}")
                continue
        
        # Fit final model with best alpha on full data
        if best_alpha is None:
            print(f"  All alphas failed, fitting with default alpha=0.01")
            best_alpha = 0.01

        full_key = (model_key, float(best_alpha), 'full')
        model = make_model(best_alpha)
        model.fit(
            X, T, E, weights=weights,
            initial_coef=self._cox_warm_starts.get(full_key),
        )
        self._cox_warm_starts[full_key] = model.coef_.copy()
        model.oof_risk_ = best_oof_risk
        model.cv_concordance_ = best_concordance
        
        return model, best_alpha
    
    def _predict_risk(self, model, X):
        """
        Predict log-risk scores using TorchCoxPH.
        
        Returns log partial hazard (higher = higher risk).
        """
        return model.predict_log_partial_hazard(X)
    
    def _policy_outputs(self, S, stochastic=False):
        """Return action weights used for risk and soft policy probabilities."""
        logits = self.policy_network.get_logits(S)
        soft_probs = torch.softmax(logits / self.gumbel_temperature, dim=1)
        if not self.hard_policy:
            return soft_probs, soft_probs
        if stochastic:
            action_weights = F.gumbel_softmax(
                logits, tau=self.gumbel_temperature, hard=True, dim=1
            )
        else:
            actions = torch.argmax(logits, dim=1)
            action_weights = F.one_hot(
                actions, num_classes=logits.shape[1]
            ).to(dtype=logits.dtype)
        return action_weights, soft_probs

    def _get_policy_probs(self, S, hard=False):
        """
        Get policy probabilities for state vectors.
        
        Parameters
        ----------
        S : ndarray
            State matrix (n_samples, 4)
            
        Returns
        -------
        probs : ndarray
            Policy probabilities (n_samples, 3)
        """
        self.policy_network.eval()
        with torch.no_grad():
            S_tensor = torch.as_tensor(
                np.ascontiguousarray(S), dtype=torch.float32, device=self.device
            )
            action_weights, soft_probs = self._policy_outputs(
                S_tensor, stochastic=False
            )
            output = action_weights if hard else soft_probs
            return output.detach().cpu().numpy()
    
    def _train_policy_epoch(self, S, R, P, RP, E, T, rp_cost=0.0,
                            exploration_prior=None):
        """Train one full-risk-set policy epoch."""
        self.policy_network.train()
        action_weights, soft_probs = self._policy_outputs(S, stochastic=True)
        base_loss = self.policy_loss_fn(
            action_weights, R, P, RP, E, T,
            regularization_probs=soft_probs,
            exploration_prior=exploration_prior,
        )
        rp_penalty = (
            self.rp_cost_weight * rp_cost * action_weights[:, 2].mean()
        )
        loss = base_loss + rp_penalty

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update stochastic/adaptive schedules from training only. Validation
        # and objective reporting must be side-effect free.
        if hasattr(self.policy_loss_fn, 'update_exploration_weight'):
            self.policy_loss_fn.update_exploration_weight(loss.detach())
        if hasattr(self.policy_loss_fn, 'update_parameters'):
            self.policy_loss_fn.update_parameters()
        if self.hard_policy:
            self.gumbel_temperature = max(
                self.gumbel_min_temperature,
                self.gumbel_temperature * self.gumbel_anneal_rate,
            )

        return loss.item()

    def _fit_policy_network(self, S, R, P, RP, E, T, train_idx, select_idx,
                            rp_cost, verbose=True):
        """Fit policy with a shared selection set and restore one checkpoint."""
        train_tensors = [
            torch.as_tensor(np.ascontiguousarray(values[train_idx]),
                            dtype=torch.float32, device=self.device)
            for values in (S, R, P, RP, E, T)
        ]
        select_tensors = [
            torch.as_tensor(np.ascontiguousarray(values[select_idx]),
                            dtype=torch.float32, device=self.device)
            for values in (S, R, P, RP, E, T)
        ]
        best_val_loss = float('inf')
        best_val_cindex = -np.inf
        best_checkpoint = None
        patience_counter = 0
        patience = 10
        for epoch in range(self.policy_epochs):
            train_loss = self._train_policy_epoch(
                *train_tensors, rp_cost=rp_cost
            )

            self.policy_network.eval()
            with torch.no_grad():
                S_select, R_select, P_select, RP_select, E_select, T_select = (
                    select_tensors
                )
                action_select, soft_select = self._policy_outputs(
                    S_select, stochastic=False
                )
                components = self.policy_loss_fn(
                    action_select, R_select, P_select, RP_select,
                    E_select, T_select, return_components=True,
                    regularization_probs=soft_select,
                )
                rp_penalty = (
                    self.rp_cost_weight * rp_cost
                    * action_select[:, 2].mean()
                )
                val_loss = (components['total_loss'] + rp_penalty).item()
                select_risk = (
                    action_select[:, 0] * R_select
                    + action_select[:, 1] * P_select
                    + action_select[:, 2] * RP_select
                )
                val_cindex = concordance_index(
                    T_select.detach().cpu().numpy(),
                    -select_risk.detach().cpu().numpy(),
                    E_select.detach().cpu().numpy().astype(bool),
                )
                exploitation_loss = components['cox_loss'].item()
                exploration_loss = (
                    components['total_loss'].item() - exploitation_loss
                )

            # Select epochs by the complete regularized objective so adequate
            # exploration is retained. C-index only breaks effectively tied
            # validation losses.
            loss_tolerance = 1e-6
            improved = (
                val_loss < best_val_loss - loss_tolerance
                or (
                    abs(val_loss - best_val_loss) <= loss_tolerance
                    and val_cindex > best_val_cindex
                )
            )
            if improved:
                best_val_loss = val_loss
                best_val_cindex = val_cindex
                best_checkpoint = {
                    'policy': {
                        key: value.detach().cpu().clone()
                        for key, value in self.policy_network.state_dict().items()
                    },
                    'optimizer': copy.deepcopy(self.policy_optimizer.state_dict()),
                    'loss_fn': copy.deepcopy(self.policy_loss_fn),
                    'gumbel_temperature': self.gumbel_temperature,
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"  Epoch {epoch + 1}/{self.policy_epochs}: "
                    f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                    f"Val C-index = {val_cindex:.4f}, "
                    f"Exploitation Loss = {exploitation_loss:.4f}, "
                    f"Exploration Loss = {exploration_loss:.4f}, "
                    f"RP Penalty = {rp_penalty.item():.4f}, "
                    f"Exploration Weight = "
                    f"{self.policy_loss_fn.exploration_weight:.4f}, "
                    f"Gumbel T = {self.gumbel_temperature:.3f}"
                )

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if best_checkpoint is None:
            raise RuntimeError("Policy training did not produce a valid checkpoint")
        self.policy_network.load_state_dict(best_checkpoint['policy'])
        self.policy_optimizer.load_state_dict(best_checkpoint['optimizer'])
        self.policy_loss_fn = best_checkpoint['loss_fn']
        self.gumbel_temperature = best_checkpoint['gumbel_temperature']
        return best_val_loss
    
    def _init_policy_network(self):
        """Initialize the policy network and optimizer."""
        self.policy_network = PolicyNetwork(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            output_dim=3,
            dropout_rate=0.1
        ).to(self.device)
        
        self._reset_policy_optimization_state()

    def _reset_policy_optimization_state(self):
        """Initialize policy optimization state once at the start of a fit."""
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.gumbel_temperature = self.gumbel_initial_temperature
        self._init_loss_function()
    
    def fit(self, X_rad, X_path, y):
        """
        Fit the Contextual Bandit with EM algorithm.
        
        Parameters
        ----------
        X_rad : ndarray
            Radiomic features
        X_path : ndarray
            Pathomic features
        y : structured array or DataFrame
            Survival data with fields 'event' and 'duration'
            
        Returns
        -------
        self : ContextualBandit
            Fitted instance
        """
        # Reset fit-specific state while retaining warm starts within this fit.
        self._cox_warm_starts = {}
        self.gumbel_temperature = self.gumbel_initial_temperature
        self.objective_history = []
        self.cindex_history = []
        self.rp_cost_history = []
        self.policies = []

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        X_rad = np.ascontiguousarray(X_rad, dtype=np.float32)
        X_path = np.ascontiguousarray(X_path, dtype=np.float32)

        # Extract survival data
        if isinstance(y, pd.DataFrame):
            T_train = y["duration"].values
            E_train = y["event"].values.astype(bool)
        else:
            T_train = y["duration"]
            E_train = y["event"].astype(bool)

        # Structured-array fields are often strided views whose strides are
        # incompatible with torch.FloatTensor. Materialize compact arrays once
        # before policy and Cox training.
        T_train = np.ascontiguousarray(T_train, dtype=np.float32)
        E_train = np.ascontiguousarray(E_train, dtype=np.float32)
        
        N_train = len(T_train)

        # Use one fixed 80/20 split. The shared selection subset controls
        # policy early stopping, EM checkpointing, and expert fallback; the
        # external test fold remains untouched for final evaluation.
        indices = np.arange(N_train)
        policy_train_idx, policy_select_idx = train_test_split(
            indices, test_size=0.2, random_state=self.random_state
        )
        
        # ============================================================
        # STEP 1: INITIALIZATION - Train Global Cox Models
        # ============================================================
        
        print("Initializing global Cox models...")
        print(f"Training Radiomic model...")
        self.cox_rad, _ = self.train_survival_model(
            X_rad, T_train, E_train, alpha_range=self.alpha_range,
            model_key='radiomics'
        )
        R_train = self._predict_risk(self.cox_rad, X_rad)
        
        print(f"Training Pathomic model...")
        self.cox_path, _ = self.train_survival_model(
            X_path, T_train, E_train, alpha_range=self.alpha_range,
            model_key='pathomics'
        )
        P_train = self._predict_risk(self.cox_path, X_path)
        
        print(f"Training Radiopathomics model...")
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        self.cox_rp, _ = self.train_survival_model(
            X_rp, T_train, E_train, alpha_range=self.alpha_range,
            model_key='radiopathomics'
        )
        RP_train = self._predict_risk(self.cox_rp, X_rp)
        
        # Store initial models
        self.models_rad = [self.cox_rad]
        self.models_path = [self.cox_path]
        self.models_rp = [self.cox_rp]
        
        # Current risk scores
        self.R_curr = R_train.copy()
        self.P_curr = P_train.copy()
        self.RP_curr = RP_train.copy()
        
        # Initialize policy network
        self._init_policy_network()

        # One fixed split supports comparable early stopping, RP evidence, and
        # EM checkpoint selection. Expert OOF risks keep RP evidence independent
        # of the samples used to fit each corresponding Cox fold model.
        bootstrap_seed = 0 if self.random_state is None else self.random_state
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        if self.rp_cost_weight > 0 and self.rp_bootstrap_samples > 0:
            n_policy_train = len(policy_train_idx)
            fixed_bootstrap_indices = bootstrap_rng.integers(
                0, n_policy_train,
                size=(self.rp_bootstrap_samples, n_policy_train)
            )
        else:
            fixed_bootstrap_indices = None

        self.policy_train_indices_ = policy_train_idx.copy()
        self.policy_select_indices_ = policy_select_idx.copy()
        self.training_cindex_history = []
        self.validation_cindex_history = []
        best_em_checkpoint = None
        best_em_cindex = -np.inf
        no_improvement = 0
        em_patience = 2
        experts_updated_since_policy = True

        def capture_checkpoint(validation_cindex):
            return {
                'cox_rad': copy.deepcopy(self.cox_rad),
                'cox_path': copy.deepcopy(self.cox_path),
                'cox_rp': copy.deepcopy(self.cox_rp),
                'policy': {
                    key: value.detach().cpu().clone()
                    for key, value in self.policy_network.state_dict().items()
                },
                'optimizer': copy.deepcopy(self.policy_optimizer.state_dict()),
                'loss_fn': copy.deepcopy(self.policy_loss_fn),
                'policy_risk_reference': copy.deepcopy(
                    self.policy_risk_reference
                ),
                'gumbel_temperature': self.gumbel_temperature,
                'w_rad': self.w_rad.copy(),
                'w_path': self.w_path.copy(),
                'w_rp': self.w_rp.copy(),
                'validation_cindex': validation_cindex,
            }

        def prepare_rp_cost():
            if self.rp_cost_weight == 0:
                return 0.0, None
            oof_risks = (
                self.cox_rad.oof_risk_,
                self.cox_path.oof_risk_,
                self.cox_rp.oof_risk_,
            )
            if any(risk is None for risk in oof_risks):
                raise RuntimeError("OOF expert risks are required for RP cost")
            R_oof, P_oof, RP_oof = oof_risks
            if not all(np.isfinite(risk).all() for risk in oof_risks):
                raise RuntimeError("OOF expert risks contain non-finite values")
            return self._compute_rp_cost(
                R_oof[policy_train_idx], P_oof[policy_train_idx],
                RP_oof[policy_train_idx], E_train[policy_train_idx],
                T_train[policy_train_idx],
                bootstrap_indices=fixed_bootstrap_indices
            )

        def fit_aligned_policy(verbose=True):
            rp_cost, rp_cost_info = prepare_rp_cost()

            raw_oof = {
                'R': self.cox_rad.oof_risk_,
                'P': self.cox_path.oof_risk_,
                'RP': self.cox_rp.oof_risk_,
            }
            raw_full = {
                'R': self.R_curr,
                'P': self.P_curr,
                'RP': self.RP_curr,
            }
            self.policy_risk_reference = self._fit_policy_risk_reference(
                raw_full, policy_train_idx, E_train, T_train,
                reliability_risks=raw_oof
            )

            # Original hybrid construction: train the policy on current
            # full-fit risks and substitute OOF risks only on the held-out
            # policy-selection subset.
            R_policy = self._apply_policy_risk_reference(
                self.R_curr, 'R', self.policy_risk_reference
            )
            P_policy = self._apply_policy_risk_reference(
                self.P_curr, 'P', self.policy_risk_reference
            )
            RP_policy = self._apply_policy_risk_reference(
                self.RP_curr, 'RP', self.policy_risk_reference
            )
            S_policy = self._make_policy_state(R_policy, P_policy, RP_policy)
            R_for_fit = R_policy.copy()
            P_for_fit = P_policy.copy()
            RP_for_fit = RP_policy.copy()
            for name, target in (
                ('R', R_for_fit), ('P', P_for_fit), ('RP', RP_for_fit)
            ):
                transformed_oof = self._apply_policy_risk_reference(
                    raw_oof[name], name, self.policy_risk_reference
                )
                target[policy_select_idx] = transformed_oof[policy_select_idx]
            S_for_fit = self._make_policy_state(R_for_fit, P_for_fit, RP_for_fit)

            self.rp_cost_history.append(rp_cost)
            if verbose:
                if rp_cost_info is None:
                    print("RP evidence penalty: disabled")
                else:
                    print(
                        f"RP evidence cost: {rp_cost:.4f} "
                        f"(OOF C-index R={rp_cost_info['cindex_rad']:.4f}, "
                        f"P={rp_cost_info['cindex_path']:.4f}, "
                        f"RP={rp_cost_info['cindex_rp']:.4f}; "
                        f"lower gains RP-R={rp_cost_info['lower_gain_vs_rad']:.4f}, "
                        f"RP-P={rp_cost_info['lower_gain_vs_path']:.4f})"
                    )
                ref = self.policy_risk_reference
                print(
                    "Policy risk reference: "
                    "separate full-fit z-scales "
                    f"R/P/RP={ref['scales']['R']:.4f}/"
                    f"{ref['scales']['P']:.4f}/"
                    f"{ref['scales']['RP']:.4f}; "
                    f"OOF C-index R={ref['cindex']['R']:.3f}, "
                    f"P={ref['cindex']['P']:.3f}, "
                    f"RP={ref['cindex']['RP']:.3f} (diagnostic only)"
                )
            best_val_loss = self._fit_policy_network(
                S_for_fit, R_for_fit, P_for_fit, RP_for_fit, E_train, T_train,
                policy_train_idx, policy_select_idx, rp_cost,
                verbose=verbose
            )
            action_weights = self._get_policy_probs(
                S_policy, hard=self.hard_policy
            )
            soft_probs = self._get_policy_probs(S_policy, hard=False)
            action_select = self._get_policy_probs(
                S_for_fit[policy_select_idx], hard=self.hard_policy
            )
            normalized_select_risk = (
                action_select[:, 0] * R_for_fit[policy_select_idx]
                + action_select[:, 1] * P_for_fit[policy_select_idx]
                + action_select[:, 2] * RP_for_fit[policy_select_idx]
            )
            selection_cindex = concordance_index(
                T_train[policy_select_idx], -normalized_select_risk,
                E_train[policy_select_idx].astype(bool)
            )
            expert_val_cindices = np.array([
                self._risk_cindex(
                    raw_oof[name][policy_select_idx],
                    E_train[policy_select_idx], T_train[policy_select_idx]
                )
                for name in ('R', 'P', 'RP')
            ], dtype=np.float64)
            return {
                'S': S_policy,
                'R': R_policy,
                'P': P_policy,
                'RP': RP_policy,
                'probs': action_weights,
                'soft_probs': soft_probs,
                'rp_cost': rp_cost,
                'val_loss': best_val_loss,
                'val_cindex': selection_cindex,
                'expert_val_cindices': expert_val_cindices,
                'best_expert': int(np.argmax(expert_val_cindices)),
            }

        # ============================================================
        # STEP 2: EM LOOP
        # ============================================================
        
        print("\nStarting EM iterations...")
        
        for iteration in range(self.max_iterations):
            print(f"\n--- EM Iteration {iteration + 1} ---")
            print(f"Training policy network for {self.policy_epochs} epochs...")
            aligned = fit_aligned_policy(verbose=True)
            experts_updated_since_policy = False
            # Original M-step construction: deterministic soft probabilities
            # on full-fit states, followed by a small uniform floor.
            policy_probs = aligned['soft_probs']
            floor = self.min_expert_weight
            policy_probs = policy_probs * (1.0 - 3.0 * floor) + floor
            self.w_rad = policy_probs[:, 0]
            self.w_path = policy_probs[:, 1]
            self.w_rp = policy_probs[:, 2]

            # The policy is always retained. Expert C-indices remain
            # diagnostics and never veto exploration or the weighted M-step.
            selection_cindex = aligned['val_cindex']

            self.validation_cindex_history.append(aligned['val_cindex'])
            self.cindex_history.append(selection_cindex)
            self.objective_history.append(-aligned['val_loss'])
            self.policies.append({
                key: value.detach().cpu().clone()
                for key, value in self.policy_network.state_dict().items()
            })
            self.models_rad.append(self.cox_rad)
            self.models_path.append(self.cox_path)
            self.models_rp.append(self.cox_rp)
            print(f"Aligned selection C-index: {aligned['val_cindex']:.4f}")
            print(
                "Selection expert C-indices - "
                f"Rad: {aligned['expert_val_cindices'][0]:.4f}, "
                f"Path: {aligned['expert_val_cindices'][1]:.4f}, "
                f"RP: {aligned['expert_val_cindices'][2]:.4f}"
            )

            previous_best = best_em_cindex
            if selection_cindex > best_em_cindex:
                best_em_cindex = selection_cindex
                best_em_checkpoint = capture_checkpoint(selection_cindex)
            if selection_cindex > previous_best + self.convergence_threshold:
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= em_patience:
                print("EM convergence reached on fixed selection C-index")
                break

            # ============================================================
            # M-STEP: Train Weighted Cox Models
            # ============================================================
            
            print(f"Training weighted Cox models...")

            expert_fit_weights = {}
            for expert_name, policy_weights in (
                ('Rad', self.w_rad),
                ('Path', self.w_path),
                ('RP', self.w_rp),
            ):
                fit_weights, weight_info = self._prepare_expert_fit_weights(
                    policy_weights, E_train
                )
                expert_fit_weights[expert_name] = fit_weights
                print(
                    f"  {expert_name} direct policy weights: "
                    f"event ESS={weight_info['event_ess']:.1f}, "
                    f"range=[{weight_info['min_weight']:.3f}, "
                    f"{weight_info['max_weight']:.3f}]"
                )
            
            # Radiomic model with weights
            print(f"  Weighted Radiomic model...")
            self.cox_rad, _ = self.train_survival_model(
                X_rad, T_train, E_train, weights=expert_fit_weights['Rad'],
                alpha_range=self.alpha_range, model_key='radiomics'
            )
            R_new = self._predict_risk(self.cox_rad, X_rad)
            
            # Pathomic model with weights
            print(f"  Weighted Pathomic model...")
            self.cox_path, _ = self.train_survival_model(
                X_path, T_train, E_train, weights=expert_fit_weights['Path'],
                alpha_range=self.alpha_range, model_key='pathomics'
            )
            P_new = self._predict_risk(self.cox_path, X_path)
            
            # Fusion model with weights
            print(f"  Weighted Fusion (RP) model...")
            self.cox_rp, _ = self.train_survival_model(
                X_rp, T_train, E_train, weights=expert_fit_weights['RP'],
                alpha_range=self.alpha_range, model_key='radiopathomics'
            )
            RP_new = self._predict_risk(self.cox_rp, X_rp)
            
            # ============================================================
            # UPDATE RISK SCORES
            # ============================================================
            
            self.R_curr = R_new
            self.P_curr = P_new
            self.RP_curr = RP_new
            experts_updated_since_policy = True
            
            # ============================================================
            # EVALUATE AND CHECK CONVERGENCE
            # ============================================================
            
            post_mstep_oof_cindices = np.array([
                self._risk_cindex(model.oof_risk_, E_train, T_train)
                for model in (self.cox_rad, self.cox_path, self.cox_rp)
            ])
            self.training_cindex_history.append(
                float(np.max(post_mstep_oof_cindices))
            )
            print(
                "OOF expert C-indices after M-step - "
                f"Rad: {post_mstep_oof_cindices[0]:.4f}, "
                f"Path: {post_mstep_oof_cindices[1]:.4f}, "
                f"RP: {post_mstep_oof_cindices[2]:.4f}"
            )
            
            # Print mutually exclusive policy assignments. The M-step still
            # uses the complete soft weights above.
            expert_weights = np.column_stack([
                self.w_rad, self.w_path, self.w_rp
            ])
            subgroup_counts = np.bincount(
                np.argmax(expert_weights, axis=1), minlength=3
            )
            print(f"Argmax subgroup sizes - Rad: {subgroup_counts[0]}, "
                  f"Path: {subgroup_counts[1]}, RP: {subgroup_counts[2]}")
            print(f"Mean expert weights - Rad: {self.w_rad.mean():.3f}, "
                  f"Path: {self.w_path.mean():.3f}, RP: {self.w_rp.mean():.3f}")
            

        # A final E-step aligns the policy with experts produced by the last
        # M-step. It is skipped when convergence stopped before another M-step.
        if experts_updated_since_policy:
            print("\nFinal policy normalization on the last Cox experts...")
            aligned = fit_aligned_policy(verbose=False)
            final_m_step_probs = aligned['soft_probs']
            floor = self.min_expert_weight
            final_m_step_probs = (
                final_m_step_probs * (1.0 - 3.0 * floor) + floor
            )
            self.w_rad = final_m_step_probs[:, 0]
            self.w_path = final_m_step_probs[:, 1]
            self.w_rp = final_m_step_probs[:, 2]
            selection_cindex = aligned['val_cindex']
            self.validation_cindex_history.append(aligned['val_cindex'])
            self.cindex_history.append(selection_cindex)
            self.objective_history.append(-aligned['val_loss'])
            if selection_cindex > best_em_cindex:
                best_em_cindex = selection_cindex
                best_em_checkpoint = capture_checkpoint(selection_cindex)

        if best_em_checkpoint is None:
            raise RuntimeError("EM did not produce a valid synchronized checkpoint")
        self.cox_rad = best_em_checkpoint['cox_rad']
        self.cox_path = best_em_checkpoint['cox_path']
        self.cox_rp = best_em_checkpoint['cox_rp']
        self.policy_network.load_state_dict(best_em_checkpoint['policy'])
        self.policy_optimizer.load_state_dict(best_em_checkpoint['optimizer'])
        self.policy_loss_fn = best_em_checkpoint['loss_fn']
        self.policy_risk_reference = best_em_checkpoint['policy_risk_reference']
        self.gumbel_temperature = best_em_checkpoint['gumbel_temperature']
        self.w_rad = best_em_checkpoint['w_rad']
        self.w_path = best_em_checkpoint['w_path']
        self.w_rp = best_em_checkpoint['w_rp']

        print(
            f"\nEM completed. Best selected validation C-index: "
            f"{best_em_cindex:.4f}"
        )
        print(f"Aligned policy evaluations: {len(self.validation_cindex_history)}")
        
        return self
    
    def predict_risk(self, X_rad, X_path):
        """
        Predict risk scores for new patients using the learned policy.
        
        Parameters
        ----------
        X_rad : ndarray
            Radiomic features
        X_path : ndarray
            Pathomic features
            
        Returns
        -------
        risk_scores : ndarray
            Selected OOF-centred, common-scale relative risk
        actions : ndarray
            Selected actions for each patient
        probs : ndarray
            Policy probabilities for each action
        """
        # Get risk scores from each model
        R = self._predict_risk(self.cox_rad, X_rad)
        P = self._predict_risk(self.cox_path, X_path)
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        RP = self._predict_risk(self.cox_rp, X_rp)

        R, P, RP = self._normalize_expert_risks(R, P, RP)
        S = self._make_policy_state(R, P, RP)
        
        # Get policy probabilities
        probs = self._get_policy_probs(S)
        actions = np.argmax(probs, axis=1)
        
        # Compute final risk scores
        N = len(R)
        risk_scores = np.zeros(N)
        for i in range(N):
            if actions[i] == 0:
                risk_scores[i] = R[i]
            elif actions[i] == 1:
                risk_scores[i] = P[i]
            else:
                risk_scores[i] = RP[i]
        
        return risk_scores, actions, probs

    def get_subgroup_probabilities(self, X_rad, X_path):
        """
        Get soft subgroup assignment probabilities for new patients.
        """
        R = self._predict_risk(self.cox_rad, X_rad)
        P = self._predict_risk(self.cox_path, X_path)
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        RP = self._predict_risk(self.cox_rp, X_rp)
        R, P, RP = self._normalize_expert_risks(R, P, RP)
        S = self._make_policy_state(R, P, RP)
        return self._get_policy_probs(S)
    
    def get_weighted_risk(self, X_rad, X_path):
        """
        Get weighted risk scores using policy probabilities directly (soft ensemble).
        """
        R = self._predict_risk(self.cox_rad, X_rad)
        P = self._predict_risk(self.cox_path, X_path)
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        RP = self._predict_risk(self.cox_rp, X_rp)

        R, P, RP = self._normalize_expert_risks(R, P, RP)
        S = self._make_policy_state(R, P, RP)
        probs = self._get_policy_probs(S)
        
        # Weighted risk = sum(prob * risk)
        risk_scores = probs[:, 0] * R + probs[:, 1] * P + probs[:, 2] * RP
        
        return risk_scores, probs


class ContextualBanditPipeline:
    """
    Pipeline wrapper for Contextual Bandit with survival prediction.
    """
    
    def __init__(self, bandit, use_soft_ensemble=False,
                 radiomics_scaler=None, pathomics_scaler=None):
        self.bandit = bandit
        self.use_soft_ensemble = use_soft_ensemble
        self.radiomics_scaler = (
            StandardScaler() if radiomics_scaler is None else radiomics_scaler
        )
        self.pathomics_scaler = (
            StandardScaler() if pathomics_scaler is None else pathomics_scaler
        )
        self.risk_scores_ = None
        self.actions_ = None
        self.probs_ = None

    @staticmethod
    def _as_feature_matrix(X):
        values = X.values if hasattr(X, 'values') else X
        return np.ascontiguousarray(values, dtype=np.float32)

    def _transform_inputs(self, X_rad, X_path):
        if not hasattr(self.radiomics_scaler, 'mean_'):
            raise RuntimeError("ContextualBanditPipeline must be fitted first")
        X_rad = self._as_feature_matrix(X_rad)
        X_path = self._as_feature_matrix(X_path)
        return (
            np.ascontiguousarray(
                self.radiomics_scaler.transform(X_rad), dtype=np.float32
            ),
            np.ascontiguousarray(
                self.pathomics_scaler.transform(X_path), dtype=np.float32
            ),
        )
    
    def fit(self, X_rad, X_path, y):
        X_rad = self._as_feature_matrix(X_rad)
        X_path = self._as_feature_matrix(X_path)
        X_rad_scaled = np.ascontiguousarray(
            self.radiomics_scaler.fit_transform(X_rad), dtype=np.float32
        )
        X_path_scaled = np.ascontiguousarray(
            self.pathomics_scaler.fit_transform(X_path), dtype=np.float32
        )
        self.bandit.fit(X_rad_scaled, X_path_scaled, y)
        return self
    
    def transform(self, X_rad, X_path):
        X_rad, X_path = self._transform_inputs(X_rad, X_path)
        if self.use_soft_ensemble:
            risk_scores, probs = self.bandit.get_weighted_risk(X_rad, X_path)
            self.probs_ = probs
            # Diagnostic hard assignments remain available even though risk
            # prediction uses the complete soft probability distribution.
            self.actions_ = np.argmax(probs, axis=1)
        else:
            risk_scores, actions, probs = self.bandit.predict_risk(X_rad, X_path)
            self.actions_ = actions
            self.probs_ = probs
        
        self.risk_scores_ = risk_scores
        return risk_scores
    
    def fit_transform(self, X_rad, X_path, y):
        self.fit(X_rad, X_path, y)
        return self.transform(X_rad, X_path)
    
    def get_subgroup_probs(self, X_rad, X_path):
        X_rad, X_path = self._transform_inputs(X_rad, X_path)
        return self.bandit.get_subgroup_probabilities(X_rad, X_path)
    
    def get_cindex_history(self):
        return self.bandit.cindex_history
    
    def get_objective_history(self):
        return self.bandit.objective_history
