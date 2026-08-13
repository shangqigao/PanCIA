import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from analysis.a05_outcome_prediction.variational_survival_moe import (
    ConditionalVariationalSurvivalMoE,
    PiecewiseExponentialExpert,
)


class VariationalSurvivalMoETests(unittest.TestCase):
    def test_piecewise_exponential_likelihood_handles_events_and_censoring(self):
        expert = PiecewiseExponentialExpert(state_dim=1, n_intervals=2)
        with torch.no_grad():
            expert.log_baseline_hazard.fill_(0.0)
            expert.risk.weight.zero_()
        state = torch.zeros(2, 1)
        duration = torch.tensor([0.5, 1.5])
        event = torch.tensor([1.0, 0.0])
        boundaries = torch.tensor([0.0, 1.0, 2.0])

        loglik = expert.log_likelihood(state, duration, event, boundaries)

        torch.testing.assert_close(loglik, torch.tensor([-0.5, -1.5]))

    def test_responsibility_combines_gate_and_survival_evidence(self):
        gate = torch.tensor([[0.2, 0.5, 0.3]])
        likelihood = torch.tensor(
            [[0.0, np.log(2.0), np.log(0.5)]], dtype=torch.float32
        )

        responsibility = ConditionalVariationalSurvivalMoE.posterior_responsibilities(
            gate, likelihood
        )
        expected = torch.tensor([[0.2, 1.0, 0.15]])
        expected = expected / expected.sum(dim=1, keepdim=True)

        torch.testing.assert_close(responsibility, expected)
        self.assertFalse(responsibility.requires_grad)

    def test_router_prior_kl_is_zero_when_gate_matches_prior(self):
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, state_dim=2, hidden_dim=4,
            n_intervals=2, reliability_prior=(0.2, 0.6, 0.2), device="cpu"
        )
        gate = model.reliability_prior.unsqueeze(0).repeat(3, 1)
        kl = torch.mean(torch.sum(
            gate * (torch.log(gate) - torch.log(model.reliability_prior)), dim=1
        ))
        self.assertAlmostEqual(float(kl), 0.0, places=7)

    def test_fit_and_single_patient_prediction_do_not_require_outcome(self):
        rng = np.random.default_rng(12)
        n = 48
        x_rad = rng.normal(size=(n, 3)).astype(np.float32)
        x_path = rng.normal(size=(n, 4)).astype(np.float32)
        signal = 0.7 * x_path[:, 0] - 0.3 * x_rad[:, 0]
        duration = np.exp(-signal + rng.normal(scale=0.3, size=n)).astype(np.float32)
        event = (rng.random(n) < 0.8)
        y = np.empty(n, dtype=[("event", "?"), ("duration", "<f4")])
        y["event"], y["duration"] = event, duration
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=3, path_dim=4, state_dim=2, hidden_dim=6,
            n_intervals=3, max_epochs=4, patience=3,
            mc_test_samples=3, device="cpu", random_state=4,
        ).fit(x_rad, x_path, y)

        risk, action, probs, uncertainty = model.predict(
            x_rad[:1], x_path[:1], hard=True
        )

        self.assertEqual(risk.shape, (1,))
        self.assertEqual(action.shape, (1,))
        self.assertEqual(probs.shape, (1, 3))
        self.assertEqual(uncertainty.shape, (1,))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(risk).all())


if __name__ == "__main__":
    unittest.main()
