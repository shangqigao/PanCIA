"""Regression test for numerical stability of the policy Cox objective."""

import ast
from pathlib import Path
import unittest

import torch
import torch.nn as nn


def _load_weighted_cox_loss():
    source_path = (
        Path(__file__).parents[1]
        / "analysis"
        / "a05_outcome_prediction"
        / "m_survival_analysis.py"
    )
    tree = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WeightedCoxPLLoss"
    )
    namespace = {"torch": torch, "nn": nn}
    exec(
        compile(ast.Module(body=[class_node], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["WeightedCoxPLLoss"]


WeightedCoxPLLoss = _load_weighted_cox_loss()


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


if __name__ == "__main__":
    unittest.main()
