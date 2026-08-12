"""Regression test for numerical stability of the policy Cox objective."""

import ast
from pathlib import Path
import unittest

import torch
import torch.nn as nn


def _load_policy_cox_losses():
    source_path = (
        Path(__file__).parents[1]
        / "analysis"
        / "a05_outcome_prediction"
        / "m_survival_analysis.py"
    )
    tree = ast.parse(source_path.read_text())
    class_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"WeightedCoxPLLoss", "AdaptiveWeightedCoxPLLoss"}
    ]
    namespace = {"torch": torch, "nn": nn}
    exec(
        compile(ast.Module(body=class_nodes, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["WeightedCoxPLLoss"], namespace["AdaptiveWeightedCoxPLLoss"]


WeightedCoxPLLoss, AdaptiveWeightedCoxPLLoss = _load_policy_cox_losses()


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

    def test_adaptive_loss_has_no_risk_variance_penalty(self):
        loss_fn = AdaptiveWeightedCoxPLLoss(initial_exploration_weight=0.2)
        probs = torch.full((4, 3), 1.0 / 3.0)
        risk = torch.tensor([0.2, -0.1, 0.5, 0.0])
        events = torch.tensor([1.0, 0.0, 1.0, 1.0])
        times = torch.tensor([4.0, 3.0, 2.0, 1.0])

        components = loss_fn(
            probs, risk, -risk, 3.0 * risk, events, times,
            return_components=True,
        )

        self.assertNotIn("uncertainty_bonus", components)
        self.assertNotIn("uncertainty_value", components)
        torch.testing.assert_close(
            components["total_loss"],
            components["cox_loss"]
            + components["entropy_penalty"]
            + components["coverage_penalty"],
        )

    def test_adaptive_exploration_uses_floors_not_uniformity_rewards(self):
        loss_fn = AdaptiveWeightedCoxPLLoss(
            initial_exploration_weight=0.2,
            target_entropy_fraction=0.7,
            min_action_coverage=0.1,
        )
        risk = torch.tensor([0.2, -0.1, 0.5, 0.0])
        events = torch.tensor([1.0, 0.0, 1.0, 1.0])
        times = torch.tensor([4.0, 3.0, 2.0, 1.0])
        uniform = torch.full((4, 3), 1.0 / 3.0)
        collapsed = torch.tensor([[0.98, 0.01, 0.01]]).repeat(4, 1)

        uniform_parts = loss_fn(
            uniform, risk, risk, risk, events, times,
            return_components=True,
        )
        collapsed_parts = loss_fn(
            collapsed, risk, risk, risk, events, times,
            return_components=True,
        )

        self.assertEqual(uniform_parts["entropy_penalty"].item(), 0.0)
        self.assertEqual(uniform_parts["coverage_penalty"].item(), 0.0)
        self.assertGreater(collapsed_parts["entropy_penalty"].item(), 0.0)
        self.assertGreater(collapsed_parts["coverage_penalty"].item(), 0.0)

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
