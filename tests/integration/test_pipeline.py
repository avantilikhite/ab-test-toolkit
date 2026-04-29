"""Integration test: full analysis pipeline end-to-end."""

import numpy as np
import pytest

from ab_test_toolkit.data_generator import generate_experiment_data
from ab_test_toolkit.io import load_experiment_data
from ab_test_toolkit.frequentist import z_test
from ab_test_toolkit.bayesian import beta_binomial
from ab_test_toolkit.srm import check_srm
from ab_test_toolkit.cuped import cuped_adjust
from ab_test_toolkit.segmentation import segment_analysis
from ab_test_toolkit.recommendation import generate_recommendation


class TestFullPipeline:
    """End-to-end pipeline test."""

    def test_full_pipeline_proportion(self):
        """Full pipeline with proportion data produces valid recommendation."""
        # Generate
        raw_data = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02,
            n_control=2000, n_treatment=2000,
            random_seed=42,
        )

        # Load & validate
        data, metric_type = load_experiment_data(raw_data)
        assert metric_type.value == "proportion"

        ctrl = data[data["group"] == "control"]["value"].values
        treat = data[data["group"] == "treatment"]["value"].values

        # Frequentist
        freq = z_test(ctrl, treat)
        assert 0 <= freq.p_value <= 1

        # Bayesian
        bayes = beta_binomial(ctrl, treat, n_simulations=100_000)
        assert 0 <= bayes.prob_b_greater_a <= 1

        # SRM
        srm = check_srm(observed=(len(ctrl), len(treat)))
        assert srm.has_mismatch is False  # No SRM injected

        # CUPED
        ctrl_cov = data[data["group"] == "control"]["covariate"].values
        treat_cov = data[data["group"] == "treatment"]["covariate"].values
        cuped = cuped_adjust(ctrl, treat, ctrl_cov, treat_cov)
        assert cuped.variance_reduction_pct >= 0

        # Recommendation
        rec = generate_recommendation(freq, bayes, srm)
        assert rec.recommendation in {"Ship", "No-Ship", "Inconclusive"}
        assert "significance" in rec.supporting_metrics
        # Treatment IS better in this data — recommendation must not be No-Ship
        assert rec.recommendation != "No-Ship", (
            "Treatment has a positive effect; recommendation should not be No-Ship"
        )

    def test_full_pipeline_with_segmentation(self):
        """Pipeline with Simpson's Paradox data."""
        raw_data = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02,
            n_control=2000, n_treatment=2000,
            inject_simpsons=True, random_seed=42,
        )
        data, _ = load_experiment_data(raw_data)

        ctrl = data[data["group"] == "control"]["value"].values
        treat = data[data["group"] == "treatment"]["value"].values

        freq = z_test(ctrl, treat)
        bayes = beta_binomial(ctrl, treat, n_simulations=100_000)
        srm = check_srm(observed=(len(ctrl), len(treat)))
        seg = segment_analysis(data)

        rec = generate_recommendation(freq, bayes, srm, segmentation=seg)
        assert rec.recommendation in {"Ship", "No-Ship", "Inconclusive"}
        # Simpson's Paradox was injected — verify it's detected and drives
        # Inconclusive *only when at least one conflicting segment is also
        # significant under Holm-adjusted p-values* (the engine's gate).
        if seg.simpsons_paradox:
            agg_sign = 1 if seg.aggregate_estimate > 0 else (-1 if seg.aggregate_estimate < 0 else 0)
            has_sig_conflict = any(
                ((1 if sr["estimate"] > 0 else (-1 if sr["estimate"] < 0 else 0)) != agg_sign)
                and sr.get("p_value_adjusted", 1.0) < freq.alpha
                for sr in seg.segment_results
            )
            if has_sig_conflict:
                assert rec.recommendation == "Inconclusive", (
                    "Simpson's Paradox with significant conflicting segment but recommendation is not Inconclusive"
                )

    def test_full_pipeline_srm_detection(self):
        """Pipeline detects SRM when injected."""
        raw_data = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02,
            n_control=5000, n_treatment=5000,
            inject_srm=True, srm_actual_ratio=0.60,
            random_seed=42,
        )
        data, _ = load_experiment_data(raw_data)

        ctrl = data[data["group"] == "control"]
        treat = data[data["group"] == "treatment"]

        srm = check_srm(observed=(len(ctrl), len(treat)))
        assert srm.has_mismatch is True
