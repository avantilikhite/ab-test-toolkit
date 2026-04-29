"""Visualization module — 7 Plotly chart types for A/B test analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from ab_test_toolkit.bayesian import BayesianResult
from ab_test_toolkit.frequentist import FrequentistResult
from ab_test_toolkit.power import required_sample_size
from ab_test_toolkit.segmentation import SegmentResult

_LAYOUT = dict(template="plotly_white", height=400)


def mde_vs_n_curve(power_data: pd.DataFrame) -> go.Figure:
    """Line chart: MDE vs required sample size."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=power_data["mde"],
        y=power_data["n_total"],
        mode="lines+markers",
        name="Sample Size",
    ))
    fig.update_layout(
        title="MDE vs Required Sample Size",
        xaxis_title="Minimum Detectable Effect (MDE)",
        yaxis_title="Required Total Sample Size (n)",
        **_LAYOUT,
    )
    return fig


def power_loss_curve(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> go.Figure:
    """Line chart: extra sample required (vs balanced) at varying allocation ratios.

    The function name is retained for backward compatibility, but the plotted
    quantity is **sample-size inflation vs a balanced 50/50 design** — not a
    fixed-N power loss.  Required sample size grows quickly as allocation
    becomes more skewed, holding the target MDE and power constant.
    """
    ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    losses = []
    for r in ratios:
        result = required_sample_size(
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            power=power,
            allocation_ratio=r,
        )
        losses.append(result.sample_inflation_pct)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratios,
        y=losses,
        mode="lines+markers",
        name="Sample Inflation %",
    ))
    fig.update_layout(
        title="Sample-Size Inflation vs Allocation Ratio (target MDE & power held fixed)",
        xaxis_title="Allocation Ratio (treatment / control)",
        yaxis_title="Extra sample required vs balanced (%)",
        **_LAYOUT,
    )
    return fig


def ci_comparison_plot(
    result: FrequentistResult,
    control_label: str = "Control",
    treatment_label: str = "Treatment",
) -> go.Figure:
    """Error bar chart showing treatment effect with confidence interval."""
    ci_half = (result.ci_upper - result.ci_lower) / 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[result.point_estimate],
        y=["Effect"],
        error_x=dict(type="constant", value=ci_half),
        mode="markers",
        marker=dict(size=10),
        name="Treatment Effect",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="grey", annotation_text="Zero")
    fig.update_layout(
        title="Confidence Interval for Treatment Effect",
        xaxis_title="Effect Estimate",
        **_LAYOUT,
    )
    return fig


def posterior_plot(result: BayesianResult) -> go.Figure:
    """Distribution curves for control and treatment posteriors."""
    fig = go.Figure()

    if result.model_type == "beta_binomial":
        ctrl = result.control_posterior
        treat = result.treatment_posterior
        x_min = min(
            stats.beta.ppf(0.001, ctrl["alpha"], ctrl["beta"]),
            stats.beta.ppf(0.001, treat["alpha"], treat["beta"]),
        )
        x_max = max(
            stats.beta.ppf(0.999, ctrl["alpha"], ctrl["beta"]),
            stats.beta.ppf(0.999, treat["alpha"], treat["beta"]),
        )
        x = np.linspace(x_min, x_max, 500)
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, ctrl["alpha"], ctrl["beta"]),
            mode="lines", name="Control",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, treat["alpha"], treat["beta"]),
            mode="lines", name="Treatment",
        ))
    else:
        # Student-t marginal posterior (schema: mean, scale, df)
        ctrl = result.control_posterior
        treat = result.treatment_posterior
        ctrl_scale = float(ctrl.get("scale", 0.0))
        treat_scale = float(treat.get("scale", 0.0))
        ctrl_df = float(ctrl.get("df", 1.0))
        treat_df = float(treat.get("df", 1.0))
        # Guard: degenerate posterior (zero scale)
        if ctrl_scale < 1e-12 and treat_scale < 1e-12:
            fig.add_annotation(
                text="Degenerate posterior — both groups have zero variance",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14),
            )
        else:
            _scale_min = max(ctrl_scale, treat_scale, 1e-12)
            x_min = min(
                ctrl["mean"] - 4 * _scale_min if ctrl_scale < 1e-12 else stats.t.ppf(0.001, ctrl_df, loc=ctrl["mean"], scale=ctrl_scale),
                treat["mean"] - 4 * _scale_min if treat_scale < 1e-12 else stats.t.ppf(0.001, treat_df, loc=treat["mean"], scale=treat_scale),
            )
            x_max = max(
                ctrl["mean"] + 4 * _scale_min if ctrl_scale < 1e-12 else stats.t.ppf(0.999, ctrl_df, loc=ctrl["mean"], scale=ctrl_scale),
                treat["mean"] + 4 * _scale_min if treat_scale < 1e-12 else stats.t.ppf(0.999, treat_df, loc=treat["mean"], scale=treat_scale),
            )
            x = np.linspace(x_min, x_max, 500)
            if ctrl_scale >= 1e-12:
                fig.add_trace(go.Scatter(
                    x=x, y=stats.t.pdf(x, ctrl_df, loc=ctrl["mean"], scale=ctrl_scale),
                    mode="lines", name="Control",
                ))
            else:
                fig.add_vline(x=ctrl["mean"], line_dash="dash", annotation_text="Control (point mass)")
            if treat_scale >= 1e-12:
                fig.add_trace(go.Scatter(
                    x=x, y=stats.t.pdf(x, treat_df, loc=treat["mean"], scale=treat_scale),
                    mode="lines", name="Treatment",
                ))
            else:
                fig.add_vline(x=treat["mean"], line_dash="dash", annotation_text="Treatment (point mass)")

    fig.add_annotation(
        text=f"P(B>A) = {result.prob_b_greater_a:.2%}",
        xref="paper", yref="paper", x=0.95, y=0.95,
        showarrow=False, font=dict(size=14),
    )
    _x_label = "Rate" if result.model_type == "beta_binomial" else "Metric value"
    fig.update_layout(
        title="Posterior Distributions",
        xaxis_title=_x_label,
        yaxis_title="Density",
        **_LAYOUT,
    )
    return fig


def segment_comparison_chart(result: SegmentResult) -> go.Figure:
    """Bar chart with per-segment treatment effects and error bars."""
    segments = [s["segment"] for s in result.segment_results]
    estimates = [s["estimate"] for s in result.segment_results]
    ci_lowers = [s["ci"][0] for s in result.segment_results]
    ci_uppers = [s["ci"][1] for s in result.segment_results]
    errors = [(est - lo, hi - est) for est, lo, hi in zip(estimates, ci_lowers, ci_uppers)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=segments,
        y=estimates,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[e[1] for e in errors],
            arrayminus=[e[0] for e in errors],
        ),
        name="Segment Effect",
    ))
    fig.add_hline(
        y=result.aggregate_estimate,
        line_dash="dash", line_color="red",
        annotation_text="Aggregate",
    )
    if result.simpsons_paradox:
        fig.add_annotation(
            text="⚠️ Simpson's Paradox detected",
            xref="paper", yref="paper", x=0.5, y=1.05,
            showarrow=False, font=dict(size=13, color="red"),
        )
    fig.update_layout(
        title="Segment Comparison",
        xaxis_title="Segment",
        yaxis_title="Treatment Effect",
        **_LAYOUT,
    )
    return fig


def cumulative_lift_chart(data: pd.DataFrame) -> go.Figure:
    """Running cumulative lift between treatment and control.

    If a ``day`` column exists, computes per-day means then cumulative
    averages in chronological order.  Otherwise uses row order (assumed random).
    """
    if "day" in data.columns:
        # Per-day means → cumulative averages (avoids interleaving bias)
        daily = data.groupby(["day", "group"])["value"].mean().unstack("group")
        daily = daily.sort_index()
        cum_ctrl = daily["control"].expanding().mean()
        cum_treat = daily["treatment"].expanding().mean()
        lift = cum_treat - cum_ctrl
        idx = daily.index
        x_title = "Day"
    else:
        control = data.loc[data["group"] == "control", "value"].reset_index(drop=True)
        treatment = data.loc[data["group"] == "treatment", "value"].reset_index(drop=True)
        n = min(len(control), len(treatment))
        idx = np.arange(1, n + 1)
        cum_ctrl = control.iloc[:n].cumsum() / idx
        cum_treat = treatment.iloc[:n].cumsum() / idx
        lift = cum_treat - cum_ctrl
        x_title = "Observation Index"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=lift, mode="lines", name="Cumulative Lift",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title="Cumulative Lift Over Observations",
        xaxis_title=x_title,
        yaxis_title="Cumulative Lift",
        **_LAYOUT,
    )
    return fig


def daily_treatment_effect(data: pd.DataFrame) -> go.Figure:
    """Per-day treatment effect (treatment mean − control mean)."""
    daily = data.groupby(["day", "group"])["value"].mean().unstack("group")
    effect = daily["treatment"] - daily["control"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=effect.index, y=effect.values,
        mode="lines+markers", name="Daily Effect",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title="Daily Treatment Effect",
        xaxis_title="Day",
        yaxis_title="Treatment − Control",
        **_LAYOUT,
    )
    return fig
