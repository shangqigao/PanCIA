"""Ensure Strategies 6 and 7 remain assigned to distinct model families."""

import ast
from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).parents[1]
    / "analysis"
    / "a05_outcome_prediction"
    / "m_survival_analysis.py"
)


class StrategyModelAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text())
        cls.methods = {
            node.name: node
            for parent in tree.body
            if isinstance(parent, ast.ClassDef) and parent.name == "SurvivalAnalyzer"
            for node in parent.body
            if isinstance(node, ast.FunctionDef)
        }

    @staticmethod
    def called_names(method):
        return {
            node.func.id
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_strategy_6_uses_legacy_contextual_bandit(self):
        calls = self.called_names(self.methods["strategy_6_EM_Contextual_Bandit"])
        self.assertIn("ContextualBandit", calls)
        self.assertIn("ContextualBanditPipeline", calls)
        self.assertNotIn("ConditionalVariationalSurvivalMoE", calls)

    def test_strategy_7_uses_variational_moe(self):
        calls = self.called_names(
            self.methods["strategy_7_variational_survival_moe"]
        )
        self.assertIn("ConditionalVariationalSurvivalMoE", calls)
        self.assertIn("ConditionalVariationalSurvivalPipeline", calls)
        self.assertNotIn("ContextualBandit", calls)


if __name__ == "__main__":
    unittest.main()
