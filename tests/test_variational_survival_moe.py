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
        expert = PiecewiseExponentialExpert(representation_dim=1)
        log_baseline = torch.zeros(2)
        log_risk = torch.zeros(2, 1)
        duration = torch.tensor([0.5, 1.5])
        event = torch.tensor([1.0, 0.0])
        boundaries = torch.tensor([0.0, 1.0, 2.0])

        loglik = expert.log_likelihood(
            log_risk, duration, event, boundaries, log_baseline
        )

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

    def test_tempered_responsibility_is_smoothed_toward_prior(self):
        gate = torch.tensor([[0.99, 0.005, 0.005]])
        likelihood = torch.zeros(1, 3)
        prior = torch.tensor([0.2, 0.6, 0.2])

        responsibility = ConditionalVariationalSurvivalMoE.posterior_responsibilities(
            gate, likelihood, temperature=2.0, prior=prior, prior_mix=0.1
        )

        self.assertLess(float(responsibility[0, 0]), 0.99)
        self.assertGreaterEqual(float(responsibility[0, 1]), 0.06)
        torch.testing.assert_close(responsibility.sum(1), torch.ones(1))

    def test_bayesian_responsibility_jointly_marginalizes_matched_draws(self):
        gate_draws = torch.tensor([
            [[0.8, 0.1, 0.1]],
            [[0.2, 0.7, 0.1]],
        ])
        likelihood_draws = torch.log(torch.tensor([
            [[0.9, 0.2, 0.1]],
            [[0.1, 0.8, 0.1]],
        ]))

        result = (
            ConditionalVariationalSurvivalMoE
            .posterior_responsibilities_from_draws(
                gate_draws, likelihood_draws
            )
        )

        joint_evidence = torch.tensor([[0.37, 0.29, 0.01]])
        expected = joint_evidence / joint_evidence.sum(1, keepdim=True)
        torch.testing.assert_close(result, expected)
        self.assertFalse(result.requires_grad)

        # Separate posterior means give different evidence, demonstrating why
        # E[pi*p] must not be replaced with E[pi]E[p].
        separate = gate_draws.mean(0) * likelihood_draws.exp().mean(0)
        self.assertFalse(torch.allclose(
            expected, separate / separate.sum(1, keepdim=True)
        ))

    def test_bayesian_responsibility_rejects_unmatched_draws(self):
        with self.assertRaises(ValueError):
            ConditionalVariationalSurvivalMoE.posterior_responsibilities_from_draws(
                torch.full((2, 1, 3), 1.0 / 3.0),
                torch.zeros(3, 1, 3),
            )

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

    def test_cv_prior_prefers_simpler_expert_when_rp_gain_is_unstable(self):
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, hidden_dim=4, n_intervals=2, device="cpu"
        )
        repeat_cindices = np.asarray([
            [0.60, 0.66, 0.66],
            [0.61, 0.67, 0.64],
            [0.59, 0.65, 0.67],
            [0.60, 0.66, 0.65],
            [0.60, 0.66, 0.66],
        ])

        prior, diagnostics = model._build_cv_reliability_prior(
            repeat_cindices
        )

        self.assertEqual(diagnostics["best_unimodal"], "P")
        self.assertLess(diagnostics["rp_conservative_gain"], 0.0)
        self.assertGreater(float(prior[1]), float(prior[2]))
        torch.testing.assert_close(prior.sum(), torch.tensor(1.0))

    def test_cv_prior_supports_rp_after_stable_incremental_gain(self):
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, hidden_dim=4, n_intervals=2, device="cpu"
        )
        repeat_cindices = np.asarray([
            [0.60, 0.65, 0.68],
            [0.61, 0.66, 0.69],
            [0.59, 0.64, 0.67],
            [0.60, 0.65, 0.68],
            [0.61, 0.66, 0.69],
        ])

        prior, diagnostics = model._build_cv_reliability_prior(
            repeat_cindices
        )

        self.assertGreater(diagnostics["rp_conservative_gain"], 0.02)
        self.assertGreater(diagnostics["rp_fusion_eligibility"], 0.8)
        self.assertEqual(int(torch.argmax(prior)), 2)
        torch.testing.assert_close(prior.sum(), torch.tensor(1.0))

    def test_assignment_evidence_is_centered_convex_combination(self):
        hmc = torch.tensor([[-10.0, -12.0, -14.0]])
        cv = torch.tensor([[-2.0, -1.0, -3.0]])
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, hidden_dim=4, n_intervals=2,
            cv_likelihood_weight=0.25, device="cpu",
        )

        combined = model._combine_assignment_log_likelihood(hmc, cv)
        expected = (
            0.75 * (hmc - hmc.mean(1, keepdim=True))
            + 0.25 * (cv - cv.mean(1, keepdim=True))
        )

        torch.testing.assert_close(combined, expected)
        torch.testing.assert_close(combined.mean(1), torch.zeros(1))

    def test_router_state_has_twelve_interpretable_terms(self):
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, hidden_dim=4, n_intervals=2, device="cpu"
        )
        risks = {
            "R": torch.tensor([[1.0]]),
            "P": torch.tensor([[3.0]]),
            "RP": torch.tensor([[2.0]]),
        }
        hmc_uncertainties = {
            name: torch.tensor([[value]])
            for name, value in zip(risks, [1.0, 1.0, 1.0])
        }
        cv_uncertainties = torch.full((1, 3), 2.0)

        state = model._make_router_state(
            risks, hmc_uncertainties, cv_uncertainties
        )

        scale = np.sqrt(10.0)
        expected = torch.tensor([[1.0, 3.0, 2.0, 1.0, 1.0, 1.0,
                                  2.0, 2.0, 2.0,
                                  2.0 / scale, 1.0 / scale, 1.0 / scale]],
                                dtype=torch.float32)
        torch.testing.assert_close(state, expected)

    def test_representation_standardization_preserves_relative_log_risk(self):
        torch.manual_seed(3)
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, hidden_dim=4, n_intervals=2,
            mcmc_samples=2, mcmc_chains=2, device="cpu", verbose=False,
        )
        representations = {
            name: torch.randn(12, 4) * torch.tensor([1.0, 2.0, 0.5, 3.0])
            + torch.tensor([2.0, -1.0, 4.0, 0.5])
            for name in model.expert_names
        }
        with torch.no_grad():
            for expert in model.experts.values():
                expert.risk_coef.copy_(torch.randn(4))
            old_log_risks = {
                name: model.experts[name].log_risk(representations[name]).clone()
                for name in model.expert_names
            }
            old_baseline = model.shared_log_baseline_hazard.clone()

        normalized = model._fit_representation_normalization(representations)

        for name in model.expert_names:
            new_log_risk = model.experts[name].log_risk(normalized[name])
            old_centered = old_log_risks[name] - old_log_risks[name].mean()
            torch.testing.assert_close(new_log_risk, old_centered)
            torch.testing.assert_close(
                normalized[name].mean(0), torch.zeros(4), atol=1e-6, rtol=0
            )
            torch.testing.assert_close(
                normalized[name].std(0, unbiased=False), torch.ones(4),
                atol=1e-6, rtol=0,
            )
        torch.testing.assert_close(model.shared_log_baseline_hazard, old_baseline)

    def test_all_experts_use_one_shared_baseline_parameter(self):
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, hidden_dim=4, n_intervals=3,
            mcmc_samples=2, mcmc_chains=2, device="cpu", verbose=False,
        )

        self.assertEqual(tuple(model.shared_log_baseline_hazard.shape), (3,))
        for expert in model.experts.values():
            self.assertFalse(hasattr(expert, "log_baseline_hazard"))

    def test_baseline_prior_center_adapts_to_duration_units(self):
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=1, path_dim=1, hidden_dim=2, n_intervals=2,
            mcmc_samples=2, mcmc_chains=2, device="cpu", verbose=False,
        )
        duration = np.array([100.0, 200.0, 300.0, 400.0])
        event = np.array([1.0, 0.0, 1.0, 0.0])

        model._make_boundaries(duration, event)

        expected = np.log(event.sum() / duration.sum())
        self.assertAlmostEqual(model.baseline_prior_location_, expected)
        torch.testing.assert_close(
            model.shared_log_baseline_hazard,
            torch.full((2,), expected, dtype=torch.float32),
        )

    def test_posterior_predictive_likelihood_averages_probabilities(self):
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=1, path_dim=1, hidden_dim=1, n_intervals=1,
            mcmc_samples=2, mcmc_chains=2, device="cpu", verbose=False,
        )
        model.register_buffer("time_boundaries_", torch.tensor([0.0, 2.0]))
        model.posterior_samples_ = {
            name: {"beta": torch.tensor(
                [[0.0], [np.log(2.0)]], dtype=torch.float32
            )}
            for name in model.expert_names
        }
        model.posterior_samples_["shared_log_baseline_hazard"] = torch.zeros(2, 1)
        representations = {
            name: torch.ones(1, 1) for name in model.expert_names
        }

        result = model._posterior_predictive_log_likelihood(
            representations, torch.tensor([1.0]), torch.tensor([0.0])
        )

        # Censored likelihoods are exp(-1) and exp(-2); average probabilities
        # first, then take log. This differs from averaging log-likelihoods.
        expected = np.log((np.exp(-1.0) + np.exp(-2.0)) / 2.0)
        torch.testing.assert_close(
            result, torch.full((1, 3), expected, dtype=torch.float32)
        )

    def test_chain_diagnostics_flag_stationary_parameters(self):
        moving = torch.tensor([
            [[0.0, 2.0], [0.5, 2.0], [1.0, 2.0]],
            [[0.0, 2.0], [0.5, 2.0], [1.0, 2.0]],
        ])

        rhat, ess, stationary = (
            ConditionalVariationalSurvivalMoE._chain_diagnostics(moving)
        )

        self.assertFalse(bool(stationary[0]))
        self.assertTrue(bool(stationary[1]))
        self.assertTrue(torch.isinf(rhat[1]))
        self.assertEqual(float(ess[1]), 0.0)

    def test_probability_diagnostics_detect_uniform_policy(self):
        probabilities = np.full((5, 3), 1.0 / 3.0)

        diagnostics = ConditionalVariationalSurvivalMoE.probability_diagnostics(
            probabilities
        )

        self.assertAlmostEqual(diagnostics["normalized_entropy"], 1.0)
        self.assertAlmostEqual(diagnostics["mean_max_probability"], 1.0 / 3.0)
        self.assertAlmostEqual(diagnostics["mean_top_two_margin"], 0.0)
        self.assertAlmostEqual(diagnostics["routing_mutual_information"], 0.0)
        self.assertAlmostEqual(diagnostics["path_rp_decisiveness"], 0.0)
        self.assertAlmostEqual(diagnostics["path_rp_mutual_information"], 0.0)

    def test_probability_diagnostics_detect_balanced_personalization(self):
        probabilities = np.asarray([
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
        ])

        diagnostics = ConditionalVariationalSurvivalMoE.probability_diagnostics(
            probabilities
        )

        # Marginal P/RP usage is balanced, but every patient has a decisive,
        # different preference; mutual information must capture this.
        np.testing.assert_allclose(
            diagnostics["path_rp_mean_conditional_probabilities"],
            [0.5, 0.5], atol=1e-7,
        )
        self.assertGreater(diagnostics["path_rp_decisiveness"], 0.9)
        self.assertGreater(
            diagnostics["path_rp_normalized_mutual_information"], 0.9
        )
        self.assertAlmostEqual(
            diagnostics["fraction_path_rp_odds_between_half_and_two"], 0.0
        )

    def test_soft_risk_averages_joint_draw_level_mixture(self):
        gates = [
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
        ]
        risks = [
            torch.tensor([[10.0, 1.0, 0.0]]),
            torch.tensor([[1.0, 20.0, 0.0]]),
        ]

        result = ConditionalVariationalSurvivalMoE._joint_posterior_soft_risk(
            gates, risks
        )

        # E[pi*r] = (10 + 20)/2 = 15, whereas E[pi]E[r] = 8.
        torch.testing.assert_close(result, torch.tensor([15.0]))

    def test_router_refit_averages_loss_over_posterior_draw_states(self):
        torch.manual_seed(5)
        model = ConditionalVariationalSurvivalMoE(
            rad_dim=2, path_dim=2, hidden_dim=4, n_intervals=2,
            learning_rate=0.05, beta_router_prior=0.0,
            router_refit_epochs=30, mcmc_samples=2, mcmc_chains=2,
            device="cpu", verbose=False,
        )
        states = torch.randn(3, 6, 12)
        responsibility = torch.zeros(6, 3)
        responsibility[:3, 0] = 1.0
        responsibility[3:, 1] = 1.0

        def cross_entropy():
            with torch.no_grad():
                gate = torch.softmax(model.router(states), dim=2)
                return -torch.mean(torch.sum(
                    responsibility.unsqueeze(0)
                    * torch.log(gate.clamp_min(1e-8)), dim=2
                )).item()

        before = cross_entropy()
        model._refit_router(states, responsibility)
        after = cross_entropy()

        self.assertLess(after, before)

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
            mc_test_samples=3, mcmc_warmup=2, mcmc_samples=3,
            mcmc_leapfrog_steps=2, router_refit_epochs=2,
            cv_folds=2, cv_repeats=2, cv_epochs=1,
            hmc_max_rhat=1e9, hmc_min_ess=0.0,
            verbose=False, device="cpu", random_state=4,
        ).fit(x_rad, x_path, y)

        risk, action, probs, uncertainty, diagnostics = model.predict(
            x_rad[:1], x_path[:1], hard=True
        )

        self.assertEqual(risk.shape, (1,))
        self.assertEqual(action.shape, (1,))
        self.assertEqual(probs.shape, (1, 3))
        self.assertEqual(uncertainty.shape, (1,))
        self.assertEqual(diagnostics["expert_risks"].shape, (1, 3))
        self.assertEqual(diagnostics["cv_log_risk_sd"].shape, (1, 3))
        self.assertEqual(diagnostics["hmc_log_risk_sd"].shape, (1, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(risk).all())


if __name__ == "__main__":
    unittest.main()
