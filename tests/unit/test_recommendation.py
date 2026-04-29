"""Unit tests for ab_test_toolkit.recommendation — executive summary engine."""

import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass

from ab_test_toolkit.recommendation import Recommendation, generate_recommendation, check_novelty, NoveltyCheckResult
from ab_test_toolkit.frequentist import FrequentistResult
from ab_test_toolkit.bayesian import BayesianResult
from ab_test_toolkit.srm import SRMResult
from ab_test_toolkit.segmentation import SegmentResult


def _make_freq(is_significant=True, p_value=0.01, point_estimate=0.02, effect_size=0.1):
    """Helper to create FrequentistResult for testing."""
    return FrequentistResult(
        test_type="z_test",
        statistic=2.5,
        p_value=p_value,
        ci_lower=0.001,
        ci_upper=0.04,
        point_estimate=point_estimate,
        effect_size=effect_size,
        alpha=0.05,
        is_significant=is_significant,
        normality_check=None,
    )


def _make_bayes(prob_b_gt_a=0.97, expected_loss=0.001):
    """Helper to create BayesianResult for testing."""
    return BayesianResult(
        model_type="beta_binomial",
        prob_b_greater_a=prob_b_gt_a,
        expected_loss=expected_loss,
        control_posterior={"alpha": 101, "beta": 901},
        treatment_posterior={"alpha": 121, "beta": 881},
        credible_interval=(0.005, 0.035),
        prior_config={"alpha": 1.0, "beta": 1.0},
    )


def _make_srm(has_mismatch=False):
    """Helper to create SRMResult for testing."""
    return SRMResult(
        expected_ratio=(0.5, 0.5),
        observed_ratio=(0.50, 0.50),
        chi2_statistic=0.1,
        p_value=0.75,
        has_mismatch=has_mismatch,
    )


def _make_segment(simpsons_paradox=False):
    """Helper to create SegmentResult for testing."""
    if simpsons_paradox:
        # Provide a credible reversal: aggregate is +0.02 but a significant
        # segment goes the other way under Holm-adjusted p-values.
        seg_results = [
            {"segment": "a", "estimate": -0.05, "ci": (-0.08, -0.02), "n": 500,
             "p_value": 0.001, "p_value_adjusted": 0.002},
            {"segment": "b", "estimate": -0.03, "ci": (-0.06, 0.0), "n": 500,
             "p_value": 0.04, "p_value_adjusted": 0.04},
        ]
    else:
        seg_results = [
            {"segment": "all", "estimate": 0.02, "ci": (0.005, 0.035), "n": 1000,
             "p_value": 0.01, "p_value_adjusted": 0.01},
        ]
    return SegmentResult(
        aggregate_estimate=0.02,
        aggregate_ci=(0.005, 0.035),
        segment_results=seg_results,
        simpsons_paradox=simpsons_paradox,
        simpsons_details="Sign flip detected" if simpsons_paradox else None,
        n_segments=len(seg_results),
        multiple_comparisons_note=f"{len(seg_results)} segment(s) tested",
    )


class TestGenerateRecommendation:
    """Tests for decision state machine."""

    def test_ship_decision(self):
        """Ship: significant + prob_b_gt_a > 0.95 + positive effect."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=False),
        )
        assert isinstance(rec, Recommendation)
        assert rec.recommendation == "Ship"
        assert rec.signal_strength == "strong"
        assert rec.reason  # non-empty
        assert "agree" in rec.reason.lower() or "significant" in rec.reason.lower()

    def test_no_ship_negative_effect(self):
        """No-Ship: significant but negative effect."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=-0.03),
            bayesian=_make_bayes(prob_b_gt_a=0.03),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "No-Ship"
        assert rec.signal_strength == "strong"
        assert "negative" in rec.reason.lower() or "worse" in rec.reason.lower()

    def test_inconclusive_srm(self):
        """Inconclusive when SRM detected (overrides everything)."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=True),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "none"
        assert "sample ratio mismatch" in rec.reason.lower()
        assert any("SRM" in f or "Sample Ratio Mismatch" in f for f in rec.flags)

    def test_inconclusive_simpsons(self):
        """Inconclusive when Simpson's Paradox detected."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=False),
            segmentation=_make_segment(simpsons_paradox=True),
        )
        assert rec.recommendation == "Inconclusive"
        assert any("Simpson" in f for f in rec.flags)

    def test_inconclusive_not_significant(self):
        """Inconclusive when not statistically significant."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, point_estimate=0.005),
            bayesian=_make_bayes(prob_b_gt_a=0.55),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "none"

    def test_inconclusive_moderate_signal(self):
        """Inconclusive with moderate signal when Bayesian >= 90%."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.10, point_estimate=0.01),
            bayesian=_make_bayes(prob_b_gt_a=0.94),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "moderate"
        assert "directional" in rec.reason.lower() or "not enough" in rec.reason.lower()

    def test_inconclusive_weak_signal(self):
        """Inconclusive with weak signal when 70% <= Bayesian < 90%."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.30, point_estimate=0.005),
            bayesian=_make_bayes(prob_b_gt_a=0.75),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "weak"
        assert "noise" in rec.reason.lower() or "slight" in rec.reason.lower()

    def test_inconclusive_evidence_disagreement(self):
        """Inconclusive with moderate signal when freq significant but Bayes <= 0.95."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02, effect_size=0.1),
            bayesian=_make_bayes(prob_b_gt_a=0.90),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "moderate"
        assert "disagree" in rec.reason.lower()

    def test_twymans_law_flag(self):
        """Twyman's Law flag when effect is suspiciously large."""
        # Cohen's d/h >= 1.0 triggers the flag
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.15, effect_size=1.2),
            bayesian=_make_bayes(prob_b_gt_a=0.99),
            srm=_make_srm(has_mismatch=False),
        )
        assert any("Twyman" in f for f in rec.flags)

    def test_none_segmentation(self):
        """Works when segmentation is None."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=False),
            segmentation=None,
        )
        assert rec.recommendation == "Ship"

    def test_supporting_metrics_present(self):
        """Supporting metrics dict contains expected keys."""
        rec = generate_recommendation(
            frequentist=_make_freq(),
            bayesian=_make_bayes(),
            srm=_make_srm(),
        )
        assert "significance" in rec.supporting_metrics
        assert "p_value" in rec.supporting_metrics
        assert "effect_size" in rec.supporting_metrics
        assert "srm_status" in rec.supporting_metrics
        assert "prob_b_gt_a" in rec.supporting_metrics

    def test_novelty_flag_added(self):
        """Novelty flag appears when novelty detected."""
        novelty = NoveltyCheckResult(
            has_novelty=True, early_effect=0.06, late_effect=0.02,
            ratio=3.0, details="Early effect 3x late.",
        )
        rec = generate_recommendation(
            frequentist=_make_freq(),
            bayesian=_make_bayes(),
            srm=_make_srm(),
            novelty=novelty,
        )
        assert any("Novelty" in f or "novelty" in f.lower() for f in rec.flags)
        assert "novelty_ratio" in rec.supporting_metrics

    def test_no_novelty_no_flag(self):
        """No novelty flag when novelty not detected."""
        novelty = NoveltyCheckResult(
            has_novelty=False, early_effect=0.02, late_effect=0.02,
            ratio=1.0,
        )
        rec = generate_recommendation(
            frequentist=_make_freq(),
            bayesian=_make_bayes(),
            srm=_make_srm(),
            novelty=novelty,
        )
        assert not any("novelty" in f.lower() for f in rec.flags)


class TestCheckNovelty:
    """Tests for novelty detection function."""

    def test_detects_novelty_effect(self):
        """Detects novelty when early effect >> late effect."""
        rng = np.random.default_rng(42)
        n_per_group = 1000
        days = rng.integers(1, 15, size=n_per_group)

        # Control: flat 10% rate all days
        ctrl_vals = rng.binomial(1, 0.10, n_per_group)

        # Treatment: 30% rate early (days 1-3), 10% rate late (days 4+)
        treat_days = rng.integers(1, 15, size=n_per_group)
        treat_vals = np.array([
            rng.binomial(1, 0.30 if d <= 3 else 0.10)
            for d in treat_days
        ])

        df = pd.DataFrame({
            "group": ["control"] * n_per_group + ["treatment"] * n_per_group,
            "value": np.concatenate([ctrl_vals, treat_vals]),
            "day": np.concatenate([days, treat_days]),
        })
        result = check_novelty(df)
        assert isinstance(result, NoveltyCheckResult)
        assert result.has_novelty is True
        assert result.details is not None

    def test_no_novelty_stable_effect(self):
        """No novelty when effect is stable across time."""
        rng = np.random.default_rng(42)
        n = 2000
        days = rng.integers(1, 15, size=n)
        groups = ["control"] * (n // 2) + ["treatment"] * (n // 2)
        ctrl_vals = rng.binomial(1, 0.10, n // 2)
        treat_vals = rng.binomial(1, 0.12, n // 2)
        df = pd.DataFrame({
            "group": groups,
            "value": np.concatenate([ctrl_vals, treat_vals]),
            "day": days,
        })
        result = check_novelty(df)
        assert result.has_novelty is False

    def test_no_day_column(self):
        """Gracefully handles missing day column."""
        df = pd.DataFrame({
            "group": ["control", "treatment"],
            "value": [0, 1],
        })
        result = check_novelty(df)
        assert result.has_novelty is False
        assert "skipped" in result.details.lower()

    def test_few_days(self):
        """Gracefully handles fewer than 3 days."""
        df = pd.DataFrame({
            "group": ["control"] * 10 + ["treatment"] * 10,
            "value": [0] * 10 + [1] * 10,
            "day": [1] * 10 + [2] * 10,
        })
        result = check_novelty(df)
        assert result.has_novelty is False


class TestNextSteps:
    """Tests for next-step suggestions attached to recommendations."""

    def test_ship_clean_has_rollout_steps(self):
        """Clean Ship decision suggests gradual rollout and holdback."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02, effect_size=0.1),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Ship"
        assert len(rec.next_steps) >= 1
        assert any("rollout" in s.lower() or "holdback" in s.lower() for s in rec.next_steps)

    def test_no_ship_has_iterate_steps(self):
        """No-Ship suggests investigating segments and iterating."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=-0.03, effect_size=0.1),
            bayesian=_make_bayes(prob_b_gt_a=0.03),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "No-Ship"
        assert len(rec.next_steps) >= 1
        assert any("segment" in s.lower() or "iterate" in s.lower() or "iterating" in s.lower() for s in rec.next_steps)

    def test_inconclusive_srm_has_investigate_steps(self):
        """SRM-driven Inconclusive suggests investigating the mismatch."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=True),
        )
        assert rec.recommendation == "Inconclusive"
        assert len(rec.next_steps) >= 1
        assert any("sample ratio mismatch" in s.lower() or "srm" in s.lower() for s in rec.next_steps)

    def test_inconclusive_not_significant_warns_against_extending(self):
        """Underpowered Inconclusive (no signal) warns against extending."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.15, point_estimate=0.01, effect_size=0.05),
            bayesian=_make_bayes(prob_b_gt_a=0.55),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "none"
        assert len(rec.next_steps) >= 1
        combined = " ".join(rec.next_steps).lower()
        assert "do not extend" in combined or "design" in combined

    def test_inconclusive_moderate_signal_suggests_business_judgment(self):
        """Moderate signal Inconclusive suggests business judgment and re-design."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.10, point_estimate=0.01, effect_size=0.05),
            bayesian=_make_bayes(prob_b_gt_a=0.94),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "moderate"
        combined = " ".join(rec.next_steps).lower()
        assert "do not extend" in combined
        assert "business" in combined or "stakeholder" in combined
        assert "new experiment" in combined or "larger sample" in combined

    def test_inconclusive_weak_signal_steps(self):
        """Weak signal Inconclusive suggests re-design without business judgment framing."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.30, point_estimate=0.005, effect_size=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.75),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "weak"
        combined = " ".join(rec.next_steps).lower()
        assert "do not extend" in combined
        assert "reconsider" in combined or "mde" in combined.lower()

    def test_inconclusive_simpsons_has_segment_steps(self):
        """Simpson's Paradox Inconclusive suggests segment analysis."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=False),
            segmentation=_make_segment(simpsons_paradox=True),
        )
        assert rec.recommendation == "Inconclusive"
        assert any("segment" in s.lower() for s in rec.next_steps)

    def test_simpsons_no_frameworks_disagree_step(self):
        """Simpson's Paradox should not get 'frameworks disagree' or sequential testing steps."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=False),
            segmentation=_make_segment(simpsons_paradox=True),
        )
        combined = " ".join(rec.next_steps).lower()
        assert "disagree" not in combined, "Simpson's should not get 'frameworks disagree' step"
        assert "sequential" not in combined, "Simpson's should not get sequential testing step"

    def test_ship_with_novelty_suggests_monitoring(self):
        """Ship with novelty flag suggests monitoring for decay."""
        novelty = NoveltyCheckResult(
            has_novelty=True, early_effect=0.06, late_effect=0.02,
            ratio=3.0, details="Early effect 3x late.",
        )
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02, effect_size=0.1),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(has_mismatch=False),
            novelty=novelty,
        )
        assert rec.recommendation == "Ship"
        combined = " ".join(rec.next_steps).lower()
        assert "novelty" in combined or "monitor" in combined or "holdback" in combined

    def test_inconclusive_cuped_suggestion_when_no_covariate(self):
        """Suggests CUPED when moderate signal and no covariate provided."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.10, point_estimate=0.01, effect_size=0.05),
            bayesian=_make_bayes(prob_b_gt_a=0.92),
            srm=_make_srm(has_mismatch=False),
            has_covariate=False,
        )
        combined = " ".join(rec.next_steps).lower()
        assert "cuped" in combined

    def test_no_cuped_suggestion_when_covariate_present(self):
        """Adjusts CUPED messaging when covariate was already provided."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.10, point_estimate=0.01, effect_size=0.05),
            bayesian=_make_bayes(prob_b_gt_a=0.92),
            srm=_make_srm(has_mismatch=False),
            has_covariate=True,
        )
        combined = " ".join(rec.next_steps).lower()
        assert "cuped" in combined  # still mentioned but differently
        assert "already" in combined or "adjustment" in combined

    def test_evidence_disagreement_steps(self):
        """Frequentist significant but Bayesian unconvinced gets moderate-signal guidance."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02, effect_size=0.1),
            bayesian=_make_bayes(prob_b_gt_a=0.90),
            srm=_make_srm(has_mismatch=False),
        )
        assert rec.recommendation == "Inconclusive"
        assert rec.signal_strength == "moderate"
        # The reason explains the disagreement
        assert "disagree" in rec.reason.lower()
        # Next steps should advise not extending and suggest business judgment
        combined = " ".join(rec.next_steps).lower()
        assert "do not extend" in combined
        assert "business" in combined or "stakeholder" in combined

    def test_next_steps_is_list(self):
        """next_steps is always a list, even when empty."""
        rec = generate_recommendation(
            frequentist=_make_freq(),
            bayesian=_make_bayes(),
            srm=_make_srm(),
        )
        assert isinstance(rec.next_steps, list)


class TestPracticalSignificanceGate:
    """Tests for practical significance threshold."""

    def test_ship_blocked_when_effect_below_threshold(self):
        """Significant but trivially small effect → Inconclusive with threshold."""
        freq = _make_freq(is_significant=True, p_value=0.001, point_estimate=0.001, effect_size=0.01)
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm, practical_significance_threshold=0.005)
        assert rec.recommendation == "Inconclusive"
        assert "practical" in rec.reason.lower()

    def test_ship_allowed_when_effect_above_threshold(self):
        """Effect above threshold → Ship as normal (CI lower bound > threshold)."""
        freq = FrequentistResult(
            test_type="z_test", statistic=4.0, p_value=0.001,
            ci_lower=0.010, ci_upper=0.04, point_estimate=0.02,
            effect_size=0.1, alpha=0.05, is_significant=True,
            normality_check=None,
        )
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm, practical_significance_threshold=0.005)
        assert rec.recommendation == "Ship"

    def test_no_threshold_allows_small_effects(self):
        """Without threshold, even tiny effects can Ship."""
        freq = _make_freq(is_significant=True, p_value=0.001, point_estimate=0.001, effect_size=0.01)
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm)
        assert rec.recommendation == "Ship"


class TestTwymanRelativeLift:
    """Tests for relative lift Twyman's Law check."""

    def test_large_relative_lift_flagged(self):
        """60% relative lift triggers Twyman flag."""
        freq = _make_freq(point_estimate=0.06, effect_size=0.18)
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm, control_rate=0.10)
        twyman_flags = [f for f in rec.flags if "Twyman" in f]
        assert len(twyman_flags) > 0
        assert "relative lift" in twyman_flags[0]

    def test_moderate_relative_lift_not_flagged(self):
        """20% relative lift does NOT trigger Twyman."""
        freq = _make_freq(point_estimate=0.02, effect_size=0.06)
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm, control_rate=0.10)
        twyman_flags = [f for f in rec.flags if "Twyman" in f]
        assert len(twyman_flags) == 0

    def test_no_control_rate_falls_back_to_cohens(self):
        """Without control_rate, only Cohen's d/h triggers Twyman."""
        freq = _make_freq(point_estimate=0.06, effect_size=0.5)
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm)  # no control_rate
        twyman_flags = [f for f in rec.flags if "Twyman" in f]
        # Cohen's h = 0.5, below 1.0, and no relative check → no flag
        assert len(twyman_flags) == 0


class TestNoveltyHeuristicLabel:
    """Test that novelty flag is labeled as heuristic."""

    def test_novelty_flag_says_heuristic(self):
        freq = _make_freq()
        bayes = _make_bayes()
        srm = _make_srm()
        novelty = NoveltyCheckResult(has_novelty=True, early_effect=0.05, late_effect=0.02, ratio=2.5)
        rec = generate_recommendation(freq, bayes, srm, novelty=novelty)
        novelty_flags = [f for f in rec.flags if "ovelty" in f]
        assert len(novelty_flags) > 0
        assert "heuristic" in novelty_flags[0].lower()


class TestTwymanDowngrade:
    """Tests for Twyman's Law downgrading Ship → Inconclusive."""

    def test_twyman_downgrades_ship_to_inconclusive(self):
        """When Twyman fires on a Ship decision, it should downgrade to Inconclusive."""
        # Cohen's d/h >= 1.0 triggers Twyman; significant + high Bayes → Ship initially
        freq = _make_freq(is_significant=True, point_estimate=0.15, effect_size=1.2)
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm)
        assert rec.recommendation == "Inconclusive"
        assert any("Twyman" in f for f in rec.flags)

    def test_twyman_does_not_downgrade_non_ship(self):
        """When Twyman fires but decision is already Don't Ship, no downgrade."""
        # Not significant → Don't Ship; Twyman flag should still appear but not change decision
        freq = _make_freq(is_significant=False, p_value=0.5, point_estimate=0.15, effect_size=1.2)
        bayes = _make_bayes(prob_b_gt_a=0.40)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm)
        assert rec.recommendation != "Ship"
        assert any("Twyman" in f for f in rec.flags)


class TestCILowerBoundPracticalSig:
    """Tests for CI lower bound used in practical significance gate."""

    def test_ci_lower_below_threshold_blocks_ship(self):
        """When CI lower bound is below practical threshold, block Ship."""
        freq = FrequentistResult(
            test_type="z_test", statistic=2.5, p_value=0.001,
            ci_lower=0.002, ci_upper=0.04, point_estimate=0.02,
            effect_size=0.1, alpha=0.05, is_significant=True,
            normality_check=None,
        )
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm, practical_significance_threshold=0.005)
        assert rec.recommendation == "Inconclusive"

    def test_ci_lower_above_threshold_allows_ship(self):
        """When CI lower bound exceeds practical threshold, allow Ship."""
        freq = FrequentistResult(
            test_type="z_test", statistic=4.0, p_value=0.0001,
            ci_lower=0.010, ci_upper=0.04, point_estimate=0.025,
            effect_size=0.15, alpha=0.05, is_significant=True,
            normality_check=None,
        )
        bayes = _make_bayes(prob_b_gt_a=0.99)
        srm = _make_srm()
        rec = generate_recommendation(freq, bayes, srm, practical_significance_threshold=0.005)
        assert rec.recommendation == "Ship"


class TestPracticalSignificanceAndHarmful:
    """Tests for the No Effect (confident null) and Bayesian Likely Harmful branches."""

    def test_confident_null_returns_no_effect(self):
        """Tight CI inside ±practical_significance_threshold → No Effect (not Inconclusive)."""
        freq = FrequentistResult(
            test_type="z_test", statistic=0.4, p_value=0.69,
            ci_lower=-0.002, ci_upper=0.003, point_estimate=0.0005,
            effect_size=0.01, alpha=0.05, is_significant=False,
            normality_check=None,
        )
        bayes = _make_bayes(prob_b_gt_a=0.55)
        rec = generate_recommendation(
            freq, bayes, _make_srm(), practical_significance_threshold=0.005
        )
        assert rec.recommendation == "No Effect"
        assert rec.signal_strength == "strong"

    def test_no_practical_threshold_falls_through_to_inconclusive(self):
        """Without a practical-significance threshold the engine cannot return No Effect."""
        freq = FrequentistResult(
            test_type="z_test", statistic=0.4, p_value=0.69,
            ci_lower=-0.002, ci_upper=0.003, point_estimate=0.0005,
            effect_size=0.01, alpha=0.05, is_significant=False,
            normality_check=None,
        )
        rec = generate_recommendation(freq, _make_bayes(prob_b_gt_a=0.55), _make_srm())
        assert rec.recommendation == "Inconclusive"

    def test_bayesian_likely_harmful_no_ship(self):
        """prob_b_gt_a < 5% → No-Ship even when frequentist is non-significant."""
        freq = FrequentistResult(
            test_type="z_test", statistic=-1.2, p_value=0.23,
            ci_lower=-0.04, ci_upper=0.01, point_estimate=-0.015,
            effect_size=-0.08, alpha=0.05, is_significant=False,
            normality_check=None,
        )
        rec = generate_recommendation(freq, _make_bayes(prob_b_gt_a=0.02), _make_srm())
        assert rec.recommendation == "No-Ship"
        assert any("Bayesian" in f for f in rec.flags)

    def test_wide_ci_outside_threshold_stays_inconclusive(self):
        """When CI extends past the practical margin we cannot claim No Effect."""
        freq = FrequentistResult(
            test_type="z_test", statistic=0.4, p_value=0.69,
            ci_lower=-0.02, ci_upper=0.03, point_estimate=0.005,
            effect_size=0.04, alpha=0.05, is_significant=False,
            normality_check=None,
        )
        rec = generate_recommendation(
            freq, _make_bayes(prob_b_gt_a=0.55), _make_srm(),
            practical_significance_threshold=0.005,
        )
        assert rec.recommendation == "Inconclusive"


class TestNewDecisionFeatures:
    """Tests for loss-tolerance, ship-with-monitoring, Twyman low-baseline,
    Simpson significance gate, and pre-reg manifest features."""

    def test_loss_tolerance_downgrades_ship(self):
        bayes = BayesianResult(
            model_type="beta_binomial",
            prob_b_greater_a=0.97,
            expected_loss=0.10,
            control_posterior={"alpha": 100, "beta": 900},
            treatment_posterior={"alpha": 120, "beta": 880},
            credible_interval=(0.005, 0.035),
            prior_config={},
        )
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=bayes,
            srm=_make_srm(),
            loss_tolerance=0.01,
        )
        assert rec.recommendation == "Inconclusive"
        assert any("Expected loss" in f for f in rec.flags)

    def test_loss_tolerance_satisfied_keeps_ship(self):
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            loss_tolerance=0.01,
        )
        assert rec.recommendation == "Ship"

    def test_ship_with_monitoring_state(self):
        bayes = BayesianResult(
            model_type="beta_binomial",
            prob_b_greater_a=0.90,
            expected_loss=0.0005,
            control_posterior={"alpha": 100, "beta": 900},
            treatment_posterior={"alpha": 110, "beta": 890},
            credible_interval=(-0.001, 0.025),
            prior_config={},
        )
        freq = _make_freq(is_significant=False, p_value=0.12, point_estimate=0.012, effect_size=0.2)
        rec = generate_recommendation(
            frequentist=freq, bayesian=bayes, srm=_make_srm(),
            allow_ship_with_monitoring=True,
        )
        assert rec.recommendation == "Ship with Monitoring"

    def test_ship_with_monitoring_disabled_by_default(self):
        bayes = BayesianResult(
            model_type="beta_binomial", prob_b_greater_a=0.90, expected_loss=0.0005,
            control_posterior={"alpha": 100, "beta": 900},
            treatment_posterior={"alpha": 110, "beta": 890},
            credible_interval=(-0.001, 0.025), prior_config={},
        )
        freq = _make_freq(is_significant=False, p_value=0.12, point_estimate=0.012, effect_size=0.2)
        rec = generate_recommendation(frequentist=freq, bayesian=bayes, srm=_make_srm())
        assert rec.recommendation == "Inconclusive"

    def test_twyman_skipped_for_tiny_baseline(self):
        # 0.5% baseline with 0.4% absolute lift → 80% relative lift, but
        # baseline is below twyman_min_baseline so Twyman should not fire.
        freq = _make_freq(is_significant=True, point_estimate=0.004, effect_size=0.15)
        rec = generate_recommendation(
            frequentist=freq, bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            control_rate=0.005,
        )
        assert not any("Twyman" in f for f in rec.flags)

    def test_twyman_fires_above_baseline(self):
        freq = _make_freq(is_significant=True, point_estimate=0.06, effect_size=0.15)
        rec = generate_recommendation(
            frequentist=freq, bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            control_rate=0.10,  # 60% relative lift, above twyman_min_baseline
        )
        assert any("Twyman" in f for f in rec.flags)

    def test_simpson_with_no_significance_does_not_block(self):
        # Reversal pattern but no significant aggregate or segment p-value
        seg = SegmentResult(
            aggregate_estimate=0.001,
            aggregate_ci=(-0.005, 0.007),
            segment_results=[
                {"segment": "a", "estimate": -0.002, "ci": (-0.01, 0.006), "n": 500, "p_value": 0.6},
                {"segment": "b", "estimate": 0.003, "ci": (-0.005, 0.011), "n": 500, "p_value": 0.5},
            ],
            simpsons_paradox=True,
            simpsons_details="Reversal noted",
            n_segments=2,
            multiple_comparisons_note="2 segments",
        )
        freq = _make_freq(is_significant=False, p_value=0.4, point_estimate=0.001, effect_size=0.05)
        rec = generate_recommendation(
            frequentist=freq, bayesian=_make_bayes(prob_b_gt_a=0.55),
            srm=_make_srm(), segmentation=seg,
        )
        # Should NOT auto-Inconclusive on Simpson alone — should surface as flag
        assert any("Simpson" in f or "reversal" in f.lower() for f in rec.flags)

    def test_simpson_with_significance_blocks(self):
        seg = SegmentResult(
            aggregate_estimate=0.02,
            aggregate_ci=(0.005, 0.035),
            segment_results=[
                {"segment": "a", "estimate": -0.05, "ci": (-0.08, -0.02), "n": 500,
                 "p_value": 0.001, "p_value_adjusted": 0.002},
                {"segment": "b", "estimate": -0.03, "ci": (-0.06, 0.0), "n": 500,
                 "p_value": 0.04, "p_value_adjusted": 0.04},
            ],
            simpsons_paradox=True,
            simpsons_details="Reversal with significant segments",
            n_segments=2,
            multiple_comparisons_note="2 segments",
        )
        freq = _make_freq(is_significant=True, p_value=0.001, point_estimate=0.02, effect_size=0.15)
        rec = generate_recommendation(
            frequentist=freq, bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(), segmentation=seg,
        )
        assert rec.recommendation == "Inconclusive"

    def test_manifest_hash_attached(self):
        manifest = {
            "experiment_id": "exp-001",
            "primary_metric": "conversion",
            "alpha": 0.05,
            "mde": 0.02,
            "planned_n": 10000,
        }
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            manifest=manifest,
        )
        assert rec.manifest_hash is not None
        assert rec.manifest == manifest
        # Same manifest → same hash
        rec2 = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            manifest=dict(manifest),
        )
        assert rec.manifest_hash == rec2.manifest_hash

# --- Round-2 review fixes --------------------------------------------------


class TestRound2Fixes:
    def test_simpson_gate_only_blocks_when_conflicting_segment_significant(self):
        """If the conflicting segment is NOT significant under Holm, gate should not fire."""
        seg = SegmentResult(
            aggregate_estimate=0.02,
            aggregate_ci=(0.005, 0.035),
            segment_results=[
                # opposite-sign segments but with adjusted p above alpha
                {"segment": "a", "estimate": -0.02, "ci": (-0.05, 0.01), "n": 500,
                 "p_value": 0.06, "p_value_adjusted": 0.12},
                {"segment": "b", "estimate": -0.01, "ci": (-0.04, 0.02), "n": 500,
                 "p_value": 0.10, "p_value_adjusted": 0.20},
            ],
            simpsons_paradox=True,
            simpsons_details="Reversal but noisy",
            n_segments=2,
            multiple_comparisons_note="2",
        )
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, p_value=0.001, point_estimate=0.02, effect_size=0.1),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            segmentation=seg,
        )
        # Should NOT block — conflict is not significant under Holm
        assert rec.recommendation == "Ship"

    def test_simpson_gate_ignores_significant_same_sign_segment(self):
        """A significant segment that AGREES with the aggregate must not trigger Inconclusive."""
        seg = SegmentResult(
            aggregate_estimate=0.02,
            aggregate_ci=(0.005, 0.035),
            segment_results=[
                {"segment": "agrees", "estimate": 0.04, "ci": (0.02, 0.06), "n": 500,
                 "p_value": 0.001, "p_value_adjusted": 0.002},
                {"segment": "noise", "estimate": -0.005, "ci": (-0.02, 0.01), "n": 500,
                 "p_value": 0.4, "p_value_adjusted": 0.4},
            ],
            simpsons_paradox=True,  # forced for the test
            simpsons_details="Forced",
            n_segments=2,
            multiple_comparisons_note="2",
        )
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, p_value=0.001, point_estimate=0.02, effect_size=0.1),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            segmentation=seg,
        )
        assert rec.recommendation == "Ship"

    def test_twyman_downgrades_ship_with_monitoring(self):
        """Ship-with-Monitoring path should also be downgraded when Twyman fires."""
        # |d/h|=1.5 >= 1.0 → Twyman triggers regardless of baseline
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.06, point_estimate=0.5, effect_size=1.5),
            bayesian=_make_bayes(prob_b_gt_a=0.90, expected_loss=0.0001),
            srm=_make_srm(),
            allow_ship_with_monitoring=True,
            monitoring_prob_threshold=0.85,
        )
        assert rec.recommendation != "Ship with Monitoring"
        assert any("twyman" in f.lower() for f in rec.flags)

    def test_policy_snapshot_in_supporting_metrics(self):
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            loss_tolerance=0.001,
            allow_ship_with_monitoring=True,
            monitoring_prob_threshold=0.88,
        )
        policy = rec.supporting_metrics.get("policy")
        assert policy is not None
        assert policy["loss_tolerance"] == 0.001
        assert policy["allow_ship_with_monitoring"] is True
        assert policy["monitoring_prob_threshold"] == 0.88
        assert "alpha" in policy

    def test_manifest_hash_not_in_flags(self):
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=True, point_estimate=0.02),
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            manifest={"experiment_id": "x"},
        )
        assert rec.manifest_hash is not None
        assert not any("manifest" in f.lower() for f in rec.flags)
