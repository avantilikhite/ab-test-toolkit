"""Unit tests for ab_test_toolkit.visualization — all 7 Plotly chart types."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ab_test_toolkit.visualization import (
    mde_vs_n_curve,
    power_loss_curve,
    ci_comparison_plot,
    posterior_plot,
    segment_comparison_chart,
    cumulative_lift_chart,
    daily_treatment_effect,
)
from ab_test_toolkit.power import power_curve
from ab_test_toolkit.frequentist import FrequentistResult, z_test
from ab_test_toolkit.bayesian import BayesianResult, beta_binomial
from ab_test_toolkit.segmentation import SegmentResult


# --- US1: Power Charts ---

class TestMDEvsNCurve:
    """Tests for MDE-vs-N line chart."""

    def test_returns_figure(self):
        pc = power_curve(baseline_rate=0.10, mde_range=(0.01, 0.05))
        fig = mde_vs_n_curve(pc)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self):
        pc = power_curve(baseline_rate=0.10, mde_range=(0.01, 0.05))
        fig = mde_vs_n_curve(pc)
        assert len(fig.data) >= 1

    def test_axis_labels(self):
        pc = power_curve(baseline_rate=0.10, mde_range=(0.01, 0.05))
        fig = mde_vs_n_curve(pc)
        assert fig.layout.xaxis.title.text is not None
        assert fig.layout.yaxis.title.text is not None


class TestPowerLossCurve:
    """Tests for power loss vs allocation ratio."""

    def test_returns_figure(self):
        fig = power_loss_curve(baseline_rate=0.10, mde=0.02)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self):
        fig = power_loss_curve(baseline_rate=0.10, mde=0.02)
        assert len(fig.data) >= 1


# --- US2: CI Comparison ---

class TestCIComparisonPlot:
    """Tests for confidence interval comparison chart."""

    def test_returns_figure(self):
        result = FrequentistResult(
            test_type="z_test", statistic=2.0, p_value=0.04,
            ci_lower=-0.01, ci_upper=0.05, point_estimate=0.02,
            effect_size=0.1, alpha=0.05, is_significant=True,
            normality_check=None,
        )
        fig = ci_comparison_plot(result)
        assert isinstance(fig, go.Figure)

    def test_has_two_group_traces(self):
        result = FrequentistResult(
            test_type="z_test", statistic=2.0, p_value=0.04,
            ci_lower=-0.01, ci_upper=0.05, point_estimate=0.02,
            effect_size=0.1, alpha=0.05, is_significant=True,
            normality_check=None,
        )
        fig = ci_comparison_plot(result)
        assert len(fig.data) >= 1  # At least one trace


# --- US3: Posterior Plot ---

class TestPosteriorPlot:
    """Tests for Bayesian posterior distribution chart."""

    def test_returns_figure(self):
        result = BayesianResult(
            model_type="beta_binomial",
            prob_b_greater_a=0.95,
            expected_loss=0.001,
            control_posterior={"alpha": 101, "beta": 901},
            treatment_posterior={"alpha": 121, "beta": 881},
            credible_interval=(0.005, 0.035),
            prior_config={"alpha": 1.0, "beta": 1.0},
        )
        fig = posterior_plot(result)
        assert isinstance(fig, go.Figure)

    def test_has_two_distribution_traces(self):
        result = BayesianResult(
            model_type="beta_binomial",
            prob_b_greater_a=0.95,
            expected_loss=0.001,
            control_posterior={"alpha": 101, "beta": 901},
            treatment_posterior={"alpha": 121, "beta": 881},
            credible_interval=(0.005, 0.035),
            prior_config={"alpha": 1.0, "beta": 1.0},
        )
        fig = posterior_plot(result)
        assert len(fig.data) >= 2  # Control and treatment

    def test_normal_normal_returns_figure(self):
        """Normal-Normal posterior plot renders without crashing."""
        result = BayesianResult(
            model_type="normal_normal",
            prob_b_greater_a=0.85,
            expected_loss=0.05,
            control_posterior={"mean": 10.0, "scale": 0.2, "df": 100.0},
            treatment_posterior={"mean": 10.5, "scale": 0.22, "df": 100.0},
            credible_interval=(-0.1, 1.1),
            prior_config={"prior_mean": 10.0, "prior_variance": 4.0},
        )
        fig = posterior_plot(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_normal_normal_with_real_data(self):
        """Normal-Normal from actual computation through posterior_plot."""
        from ab_test_toolkit.bayesian import normal_normal
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 200)
        treatment = rng.normal(10.5, 2.0, 200)
        result = normal_normal(control, treatment)
        fig = posterior_plot(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


# --- US6: Segment Chart ---

class TestSegmentComparisonChart:
    """Tests for per-segment comparison chart."""

    def test_returns_figure(self):
        result = SegmentResult(
            aggregate_estimate=0.02,
            aggregate_ci=(0.005, 0.035),
            segment_results=[
                {"segment": "mobile", "estimate": 0.03, "ci": (0.01, 0.05), "n": 500, "p_value": 0.01},
                {"segment": "desktop", "estimate": 0.01, "ci": (-0.01, 0.03), "n": 500, "p_value": 0.10},
            ],
            simpsons_paradox=False,
            simpsons_details=None,
            n_segments=2,
            multiple_comparisons_note="2 segments tested",
        )
        fig = segment_comparison_chart(result)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self):
        result = SegmentResult(
            aggregate_estimate=0.02,
            aggregate_ci=(0.005, 0.035),
            segment_results=[
                {"segment": "mobile", "estimate": 0.03, "ci": (0.01, 0.05), "n": 500, "p_value": 0.01},
            ],
            simpsons_paradox=False,
            simpsons_details=None,
            n_segments=1,
            multiple_comparisons_note="1 segment tested",
        )
        fig = segment_comparison_chart(result)
        assert len(fig.data) >= 1


# --- US9: Time-series Charts ---

class TestCumulativeLiftChart:
    """Tests for cumulative lift chart."""

    def test_returns_figure(self):
        data = pd.DataFrame({
            "group": ["control"] * 100 + ["treatment"] * 100,
            "value": np.concatenate([
                np.random.default_rng(42).binomial(1, 0.10, 100),
                np.random.default_rng(42).binomial(1, 0.12, 100),
            ]),
        })
        fig = cumulative_lift_chart(data)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self):
        data = pd.DataFrame({
            "group": ["control"] * 100 + ["treatment"] * 100,
            "value": np.concatenate([
                np.random.default_rng(42).binomial(1, 0.10, 100),
                np.random.default_rng(42).binomial(1, 0.12, 100),
            ]),
        })
        fig = cumulative_lift_chart(data)
        assert len(fig.data) >= 1


class TestDailyTreatmentEffect:
    """Tests for daily treatment effect chart."""

    def test_returns_figure(self):
        rng = np.random.default_rng(42)
        days = np.repeat(range(1, 15), 100)
        groups = np.tile(["control"] * 50 + ["treatment"] * 50, 14)
        values = rng.binomial(1, 0.10, len(days)).astype(float)
        data = pd.DataFrame({"day": days, "group": groups, "value": values})
        fig = daily_treatment_effect(data)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self):
        rng = np.random.default_rng(42)
        days = np.repeat(range(1, 15), 100)
        groups = np.tile(["control"] * 50 + ["treatment"] * 50, 14)
        values = rng.binomial(1, 0.10, len(days)).astype(float)
        data = pd.DataFrame({"day": days, "group": groups, "value": values})
        fig = daily_treatment_effect(data)
        assert len(fig.data) >= 1
