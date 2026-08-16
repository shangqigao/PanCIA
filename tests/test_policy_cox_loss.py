"""Regression test for numerical stability of the policy Cox objective."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from analysis.a05_outcome_prediction.contextual_bandit import (
    AdaptiveWeightedCoxPLLoss,
    WeightedCoxPLLoss,
)


class PolicyCoxLossTests(unittest.TestCase):
    def test_loss_is_invariant_to_constant_risk_shift(self):
        loss_fn = WeightedCoxPLLoss(
            entropy_weight=0.0, uncertainty_weight=0.0, temperature=1.0
        )
        probs = torch.tensor(
            [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
            dtype=torch.float32,
        )
        R = torch.tensor([0.2, -0.1, 0.5])
        P = torch.tensor([0.4, 0.3, -0.2])
        RP = torch.tensor([-0.2, 0.6, 0.1])
        E = torch.tensor([1.0, 0.0, 1.0])
        T = torch.tensor([3.0, 2.0, 1.0])

        original = loss_fn(probs, R, P, RP, E, T, return_components=True)
        shifted = loss_fn(
            probs, R + 100.0, P + 100.0, RP + 100.0, E, T,
            return_components=True,
        )

        self.assertAlmostEqual(
            original["cox_loss"].item(), shifted["cox_loss"].item(), places=5
        )

    def test_adaptive_forward_does_not_mutate_exploration_schedule(self):
        loss_fn = AdaptiveWeightedCoxPLLoss(initial_exploration_weight=0.2)
        probs = torch.full((4, 3), 1.0 / 3.0)
        risk = torch.tensor([0.2, -0.1, 0.5, 0.0])
        events = torch.tensor([1.0, 0.0, 1.0, 1.0])
        times = torch.tensor([4.0, 3.0, 2.0, 1.0])

        loss_fn(probs, risk, risk, risk, events, times)

        self.assertEqual(loss_fn.exploration_weight, 0.2)
        self.assertEqual(loss_fn.step_count, 0)
        self.assertEqual(loss_fn.loss_history, [])

    def test_adaptive_loss_contains_historical_exploration_terms(self):
        loss_fn = AdaptiveWeightedCoxPLLoss(initial_exploration_weight=0.2)
        probs = torch.full((4, 3), 1.0 / 3.0)
        risk = torch.tensor([0.2, -0.1, 0.5, 0.0])
        events = torch.tensor([1.0, 0.0, 1.0, 1.0])
        times = torch.tensor([4.0, 3.0, 2.0, 1.0])

        components = loss_fn(
            probs, risk, -risk, 3.0 * risk, events, times,
            return_components=True,
        )

        self.assertIn("entropy_bonus", components)
        self.assertIn("diversity_bonus", components)
        self.assertIn("uncertainty_bonus", components)
        torch.testing.assert_close(
            components["total_loss"],
            components["cox_loss"]
            + components["entropy_bonus"]
            + components["diversity_bonus"]
            + components["uncertainty_bonus"],
        )

    def test_adaptive_exploration_prefers_uniform_soft_probabilities(self):
        loss_fn = AdaptiveWeightedCoxPLLoss(initial_exploration_weight=0.2)
        risk = torch.tensor([0.2, -0.1, 0.5, 0.0])
        events = torch.tensor([1.0, 0.0, 1.0, 1.0])
        times = torch.tensor([4.0, 3.0, 2.0, 1.0])
        uniform = torch.full((4, 3), 1.0 / 3.0)
        concentrated = torch.tensor([[0.98, 0.01, 0.01]]).repeat(4, 1)

        uniform_parts = loss_fn(
            uniform, risk, risk, risk, events, times,
            return_components=True,
        )
        concentrated_parts = loss_fn(
            concentrated, risk, risk, risk, events, times,
            return_components=True,
        )

        self.assertLess(
            uniform_parts["entropy_bonus"].item(),
            concentrated_parts["entropy_bonus"].item(),
        )
        self.assertLess(
            uniform_parts["diversity_bonus"].item(),
            concentrated_parts["diversity_bonus"].item(),
        )

    def test_policy_cox_loss_is_invariant_within_tied_time_groups(self):
        loss_fn = AdaptiveWeightedCoxPLLoss(initial_exploration_weight=0.0)
        probs = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        risk = torch.tensor([2.0, 0.0, -1.0])
        events = torch.tensor([1.0, 1.0, 0.0])
        times = torch.tensor([2.0, 2.0, 1.0])
        permutation = torch.tensor([1, 0, 2])

        original = loss_fn(
            probs, risk, risk, risk, events, times,
            return_components=True,
        )["cox_loss"]
        permuted = loss_fn(
            probs[permutation], risk[permutation], risk[permutation],
            risk[permutation], events[permutation], times[permutation],
            return_components=True,
        )["cox_loss"]

        torch.testing.assert_close(original, permuted)


if __name__ == "__main__":
    unittest.main()
