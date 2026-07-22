"""Focused tests for TorchCoxPH without importing the monolithic analysis module."""

import ast
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lifelines.utils import concordance_index


def _load_cox_classes():
    """Load Cox classes so optional PanCIA dependencies are not required."""
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
        and node.name in {
            "PolicyNetwork", "WeightedCoxPLLoss", "TorchCoxPH",
            "ContextualBandit", "ContextualBanditPipeline"
        }
    ]
    namespace = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "optim": optim,
        "concordance_index": concordance_index,
    }
    module = ast.Module(body=class_nodes, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return (
        namespace["TorchCoxPH"],
        namespace["ContextualBandit"],
        namespace["ContextualBanditPipeline"],
    )


TorchCoxPH, ContextualBandit, ContextualBanditPipeline = _load_cox_classes()


def _synthetic_survival(seed=7, n_samples=80, n_features=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    true_beta = np.zeros(n_features, dtype=np.float32)
    true_beta[:3] = [0.8, -0.6, 0.4]
    durations = np.exp(
        -(X @ true_beta) + rng.normal(scale=0.5, size=n_samples)
    )
    durations = np.round(durations, 1).astype(np.float32)
    events = (rng.random(n_samples) < 0.75).astype(np.float32)
    return X, durations, events, true_beta


class TorchCoxPHTests(unittest.TestCase):
    def test_soft_pipeline_exposes_argmax_actions(self):
        class FakeBandit:
            @staticmethod
            def get_weighted_risk(X_rad, X_path):
                probs = np.array([[0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
                return np.array([0.4, -0.2]), probs

        pipeline = ContextualBanditPipeline(FakeBandit(), use_soft_ensemble=True)
        risk = pipeline.transform(np.zeros((2, 1)), np.zeros((2, 1)))

        np.testing.assert_array_equal(risk, np.array([0.4, -0.2]))
        np.testing.assert_array_equal(pipeline.actions_, np.array([1, 2]))

    def test_straight_through_gumbel_is_one_hot_and_has_gradients(self):
        torch.manual_seed(3)
        bandit = ContextualBandit(
            hard_policy=True,
            gumbel_temperature=1.0,
            gumbel_anneal_rate=0.9,
            loss_type="weighted",
            device="cpu",
        )
        bandit._init_policy_network()
        states = torch.randn(12, 4)
        actions, soft_probs = bandit._policy_outputs(states, stochastic=True)

        torch.testing.assert_close(actions.sum(dim=1), torch.ones(12))
        self.assertTrue(torch.all((actions == 0) | (actions == 1)).item())
        loss = (actions * torch.tensor([0.0, 1.0, 2.0])).sum()
        loss.backward()
        gradients = [
            parameter.grad for parameter in bandit.policy_network.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        torch.testing.assert_close(soft_probs.sum(dim=1), torch.ones(12))

        initial_temperature = bandit.gumbel_temperature
        risk = torch.randn(12)
        bandit._train_policy_epoch(
            states, risk, -risk, 0.5 * risk,
            torch.ones(12), torch.arange(12, 0, -1, dtype=torch.float32),
        )
        self.assertAlmostEqual(
            bandit.gumbel_temperature,
            initial_temperature * bandit.gumbel_anneal_rate,
        )

    def test_deterministic_hard_policy_matches_soft_argmax(self):
        bandit = ContextualBandit(
            hard_policy=True, loss_type="weighted", device="cpu"
        )
        bandit._init_policy_network()
        states = np.random.default_rng(4).normal(size=(10, 4)).astype(np.float32)

        soft = bandit._get_policy_probs(states, hard=False)
        hard = bandit._get_policy_probs(states, hard=True)

        np.testing.assert_array_equal(np.argmax(hard, axis=1), np.argmax(soft, axis=1))
        np.testing.assert_allclose(hard.sum(axis=1), 1.0)

    def test_version_b_policy_state_contains_risks_and_signed_contrast(self):
        rng = np.random.default_rng(9)
        bandit = ContextualBandit(device="cpu")
        R = rng.normal(size=20).astype(np.float32)
        P = rng.normal(size=20).astype(np.float32)
        RP = rng.normal(size=20).astype(np.float32)

        state = bandit._make_policy_state(R, P, RP)

        self.assertEqual(state.shape, (20, 4))
        np.testing.assert_allclose(state[:, 0], R)
        np.testing.assert_allclose(state[:, 1], P)
        np.testing.assert_allclose(state[:, 2], RP)
        np.testing.assert_allclose(state[:, 3], R - P)
        self.assertTrue(np.isfinite(state).all())

    def test_weighted_breslow_loss_with_ties_matches_manual_value(self):
        X = torch.tensor([[1.0], [2.0], [3.0]])
        T = torch.tensor([2.0, 2.0, 1.0])
        E = torch.tensor([1.0, 1.0, 0.0])
        W = torch.tensor([1.0, 2.0, 1.0])
        beta = torch.tensor([0.5])

        actual = TorchCoxPH._breslow_negative_log_likelihood(beta, X, T, E, W)
        denominator = torch.exp(torch.tensor(0.5)) + 2 * torch.exp(torch.tensor(1.0))
        expected = -(0.5 + 2 * 1.0 - 3 * torch.log(denominator)) / 3

        torch.testing.assert_close(actual, expected)

    def test_fit_recovers_risk_order_and_supports_warm_start(self):
        X, T, E, true_beta = _synthetic_survival()
        weights = np.linspace(0.2, 1.0, len(T), dtype=np.float32)
        settings = dict(
            penalizer=0.001,
            l1_ratio=0.0,
            learning_rate=0.03,
            max_epochs=400,
            tolerance=1e-7,
            patience=30,
            device="cpu",
        )

        first = TorchCoxPH(**settings).fit(X, T, E, weights=weights)
        second = TorchCoxPH(**settings).fit(
            X, T, E, weights=weights[::-1].copy(), initial_coef=first.coef_
        )

        risks = first.predict_log_partial_hazard(X)
        correlation = np.corrcoef(risks, X @ true_beta)[0, 1]
        self.assertGreater(correlation, 0.9)
        self.assertTrue(np.isfinite(first.loss_))
        self.assertTrue(np.isfinite(second.loss_))
        self.assertEqual(second.coef_.shape, first.coef_.shape)

    def test_zero_weight_samples_and_tied_times_are_finite(self):
        X, T, E, _ = _synthetic_survival(seed=11)
        weights = np.ones(len(T), dtype=np.float32)
        weights[::4] = 0.0

        model = TorchCoxPH(
            penalizer=0.01,
            l1_ratio=0.9,
            max_epochs=100,
            patience=10,
        ).fit(X, T, E, weights=weights)

        self.assertTrue(np.isfinite(model.coef_).all())
        self.assertTrue(np.isfinite(model.predict_log_partial_hazard(X)).all())

    def test_zero_weight_event_groups_do_not_create_nan(self):
        X = torch.tensor([[1.0], [0.5], [-0.2]])
        T = torch.tensor([3.0, 2.0, 1.0])
        E = torch.tensor([1.0, 0.0, 1.0])
        W = torch.tensor([0.0, 0.0, 1.0])
        beta = torch.tensor([0.3])

        loss = TorchCoxPH._breslow_negative_log_likelihood(beta, X, T, E, W)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_all_censored_or_zero_weighted_events_are_rejected(self):
        X = np.ones((4, 2), dtype=np.float32)
        T = np.arange(1, 5, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "positive weight"):
            TorchCoxPH().fit(X, T, np.zeros(4, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "positive weight"):
            TorchCoxPH().fit(
                X,
                T,
                np.ones(4, dtype=np.float32),
                weights=np.zeros(4, dtype=np.float32),
            )

    def test_prediction_rejects_wrong_feature_dimension(self):
        X, T, E, _ = _synthetic_survival(n_samples=30, n_features=3)
        model = TorchCoxPH(max_epochs=20, patience=5).fit(X, T, E)

        with self.assertRaisesRegex(ValueError, "feature dimension"):
            model.predict_log_partial_hazard(np.ones((2, 4), dtype=np.float32))

    def test_contextual_bandit_cv_refits_and_caches_warm_starts(self):
        X, T, E, _ = _synthetic_survival(seed=23, n_samples=45, n_features=4)
        bandit = ContextualBandit(
            alpha_range=[0.001, 0.01],
            cv_folds=3,
            cox_max_epochs=60,
            cox_patience=8,
            device="cpu",
        )

        first, first_alpha = bandit.train_survival_model(
            X, T, E, model_key="radiomics"
        )
        fold_keys_after_first_fit = {
            key for key in bandit._cox_warm_starts if key[2] == "fold"
        }
        weights = np.linspace(0.1, 1.0, len(T), dtype=np.float32)
        second, second_alpha = bandit.train_survival_model(
            X, T, E, weights=weights, model_key="radiomics"
        )

        self.assertIn(first_alpha, bandit.alpha_range)
        self.assertIn(second_alpha, bandit.alpha_range)
        fold_keys_after_second_fit = {
            key for key in bandit._cox_warm_starts if key[2] == "fold"
        }
        self.assertEqual(fold_keys_after_first_fit, fold_keys_after_second_fit)
        self.assertEqual(len(fold_keys_after_second_fit), 6)
        self.assertEqual(first.coef_.shape, second.coef_.shape)
        self.assertTrue(np.isfinite(second.predict_log_partial_hazard(X)).all())
        self.assertEqual(second.oof_risk_.shape, (len(T),))
        self.assertTrue(np.isfinite(second.oof_risk_).all())

    def test_contextual_bandit_accepts_strided_structured_array_fields(self):
        X, T, E, _ = _synthetic_survival(seed=31, n_samples=30, n_features=3)
        survival = np.empty(30, dtype=[("event", "?"), ("duration", "<f4")])
        survival["event"] = E.astype(bool)
        survival["duration"] = T
        self.assertNotEqual(survival["duration"].strides[0] % T.itemsize, 0)

        bandit = ContextualBandit(
            alpha_range=[0.01],
            cv_folds=3,
            cox_max_epochs=20,
            cox_patience=5,
            device="cpu",
        )
        model, _ = bandit.train_survival_model(
            X, survival["duration"], survival["event"], model_key="strided"
        )
        self.assertTrue(np.isfinite(model.predict_log_partial_hazard(X)).all())

    def test_policy_risks_are_standardized_per_expert(self):
        bandit = ContextualBandit(device="cpu")
        R, P, RP = bandit._standardize_policy_risks(
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([-5.0, 0.0, 5.0]),
        )

        for risk in (R, P, RP):
            self.assertAlmostEqual(float(risk.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(risk.std()), 1.0, places=6)

        R2, P2, RP2 = bandit._apply_policy_risk_stats(
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([-5.0, 0.0, 5.0]),
        )
        np.testing.assert_allclose(R, R2)
        np.testing.assert_allclose(P, P2)
        np.testing.assert_allclose(RP, RP2)

    def test_rp_cost_is_zero_only_for_reliable_improvement(self):
        T = np.arange(1, 21, dtype=np.float32)
        E = np.ones(20, dtype=bool)
        poor_risk = T.copy()
        strong_risk = -T
        bandit = ContextualBandit(
            rp_minimum_gain=0.01,
            rp_bootstrap_samples=100,
            random_state=9,
            device="cpu",
        )

        justified_cost, info = bandit._compute_rp_cost(
            poor_risk, poor_risk, strong_risk, E, T
        )
        unsupported_cost, _ = bandit._compute_rp_cost(
            strong_risk, poor_risk, strong_risk, E, T
        )

        self.assertEqual(justified_cost, 0.0)
        self.assertGreater(info["lower_gain_vs_rad"], 0.01)
        self.assertGreater(unsupported_cost, 0.0)

if __name__ == "__main__":
    unittest.main()
