"""🎯 Experiment Design — sample size calculator and power trade-off charts."""

import streamlit as st

from ab_test_toolkit.power import power_curve, required_sample_size, required_sample_size_continuous
from ab_test_toolkit.visualization import mde_vs_n_curve, power_loss_curve

from app_utils import (
    display_error,
    display_metric_card,
    get_alpha,
    info_callout,
    metric_row,
    page_header,
    render_header_credit,
    render_sidebar_settings,
    section_header,
)

render_header_credit()
render_sidebar_settings()
page_header("🎯", "Experiment Design", "Calculate required sample sizes and visualize statistical power trade-offs.")

# --- Metric type selector ---
section_header("Configuration", "Set your experiment parameters below", "⚙️")

metric_type = st.radio(
    "Metric type",
    options=["Proportion (conversion rate)", "Continuous (revenue, time, etc.)"],
    index=0,
    horizontal=True,
    help="Proportion: binary outcomes (converted yes/no). Continuous: numeric outcomes (revenue, session time).",
)
is_proportion = metric_type.startswith("Proportion")

with st.expander("💡 Thinking in aggregates? (total revenue, total orders, etc.)"):
    st.markdown(
        "A/B tests compare **per-user averages**, not totals. If your goal is to "
        "increase an aggregate metric, frame it as a per-user value:\n\n"
        "| Your goal | Enter as | Metric type |\n"
        "|---|---|---|\n"
        "| Increase total revenue | Revenue **per user** | Continuous |\n"
        "| Increase total orders | Orders **per user** | Continuous |\n"
        "| Increase total clicks | Click-through **rate** | Proportion |\n"
        "| Increase total signups | Signup **rate** | Proportion |\n\n"
        "Enter the **per-user baseline mean** and standard deviation below, "
        "and the toolkit handles the rest."
    )

with st.container():
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        if is_proportion:
            baseline_rate = st.number_input(
                "Baseline conversion rate",
                min_value=0.0001,
                max_value=0.9999,
                value=0.10,
                step=0.001,
                format="%.4f",
                help=(
                    "Your current conversion rate before the experiment. "
                    "Supports rare events (down to 0.01%). For very rare events, "
                    "expect very large sample sizes — consider absolute MDE rather than relative."
                ),
            )
        else:
            baseline_mean = st.number_input(
                "Baseline mean",
                value=50.0,
                step=1.0,
                format="%.2f",
                help="Control group average (e.g., average revenue per user).",
            )
            baseline_std = st.number_input(
                "Baseline standard deviation",
                min_value=0.01,
                value=15.0,
                step=0.5,
                format="%.2f",
                help="Control group standard deviation.",
            )

        mde_mode = st.radio(
            "MDE mode",
            options=["relative", "absolute"],
            index=0,
            help=(
                "**Relative** (recommended): percentage change from baseline — "
                "the industry standard for product experiments (e.g., 0.05 = 5% lift). "
                "**Absolute**: raw difference. Prefer absolute when the baseline is near "
                "0% or 100% (where relative changes become misleading), for very low-rate "
                "metrics (e.g., 0.1% error rates), or in academic/clinical settings."
            ),
        )

        if mde_mode == "relative":
            mde = st.number_input(
                "Minimum Detectable Effect (MDE) — relative",
                min_value=0.001,
                max_value=0.99,
                value=0.05,
                step=0.005,
                format="%.3f",
                help="Relative change from baseline (e.g., 0.05 = 5% lift).",
            )
        else:
            mde = st.number_input(
                "Minimum Detectable Effect (MDE) — absolute",
                min_value=0.001,
                max_value=1000.0 if not is_proportion else 0.99,
                value=0.02 if is_proportion else 2.0,
                step=0.005 if is_proportion else 0.5,
                format="%.3f" if is_proportion else "%.2f",
                help="The smallest improvement you want to reliably detect (absolute difference).",
            )

    with col_right:
        allocation_ratio = st.slider(
            "Allocation ratio (treatment / control)",
            min_value=0.1,
            max_value=4.0,
            value=1.0,
            step=0.1,
            help="1.0 = 50/50 equal split. 0.25 = 80% control / 20% treatment. 2.0 = twice as many in treatment.",
        )
        daily_traffic = st.number_input(
            "Daily traffic (visitors)",
            min_value=100,
            max_value=10_000_000,
            value=10_000,
            step=1000,
            help="Average daily visitors to estimate experiment duration.",
        )

alpha = get_alpha()

power = st.slider(
    "Statistical power (1 − β)",
    min_value=0.70,
    max_value=0.95,
    value=0.80,
    step=0.05,
    help=(
        "Power is the probability the test detects a real effect of the size you specified. "
        "0.80 (industry default) accepts a 20% chance of missing a true effect; raising power "
        "to 0.90 reduces that miss rate but requires roughly 30% more sample. Lower power saves "
        "users/time but increases the false-negative rate."
    ),
)

# --- Compute & display results ---
try:
    if is_proportion:
        result = required_sample_size(
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
            mde_mode=mde_mode,
            daily_traffic=int(daily_traffic),
        )
    else:
        effective_mde = mde * abs(baseline_mean) if mde_mode == "relative" else mde
        if effective_mde == 0:
            st.error("Baseline mean is 0 — relative MDE mode is not applicable. Use absolute MDE instead.")
            st.stop()
        result = required_sample_size_continuous(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            mde=effective_mde,
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
            daily_traffic=int(daily_traffic),
        )

    st.markdown("")
    section_header("Results", "Required sample sizes for your experiment", "📋")

    days_text = f"{result.estimated_days:,}" if result.estimated_days else "N/A"
    metric_row([
        {"icon": "🅰️", "label": "Control (n)", "value": f"{result.n_control:,}",
         "help": "Number of users needed in the Control group (A) — the group that sees the original experience with no changes."},
        {"icon": "🅱️", "label": "Treatment (n)", "value": f"{result.n_treatment:,}",
         "help": "Number of users needed in the Treatment group (B) — the group that sees the new variation you're testing."},
        {"icon": "📊", "label": "Total (n)", "value": f"{result.n_total:,}",
         "help": "Total users needed across both groups combined (Control + Treatment)."},
        {"icon": "📅", "label": "Est. Days", "value": days_text,
         "help": "Estimated number of days to run the experiment based on your daily traffic volume."},
        {"icon": "⚡", "label": "Sample Inflation", "value": f"{result.sample_inflation_pct:.1f}%",
         "help": "How many extra users (in percent) the unequal allocation costs vs a balanced 50/50 design. 0% is a perfect 50/50 split. Higher values mean you need more total users to detect the same effect at the same power — consider 1:1 if traffic is the bottleneck."},
    ])

    st.markdown("")
    days_str = f"approximately **{result.estimated_days:,} days**" if result.estimated_days else "an indeterminate duration"
    power_note = ""
    if result.sample_inflation_pct > 5:
        power_note = f" Note: the unequal allocation inflates required sample size by **{result.sample_inflation_pct:.1f}%** vs balanced — consider a 1:1 ratio if traffic is scarce."
    if mde_mode == "relative" and not is_proportion:
        mde_display = f"{mde:.1%} relative (= {effective_mde:.2f} absolute)"
    else:
        mde_display = f"<strong>{mde}</strong> {mde_mode}"
    info_callout(
        f"You need <strong>{result.n_total:,}</strong> total observations "
        f"({result.n_control:,} control + {result.n_treatment:,} treatment) "
        f"to detect a {mde_display} effect, "
        f"running for {days_str} at α={alpha}, power={power:.0%}.{power_note}",
        "info",
    )

    if result.estimated_days is not None and result.estimated_days < 7:
        st.info(
            f"📅 Statistical sample size is met in **{result.estimated_days} days**, but consider running "
            f"for at least **7 days** to capture day-of-week seasonality effects. For experiments "
            f"affecting habitual behavior, 14 days is recommended."
        )

    # --- Pre-registration manifest export ---
    st.markdown("")
    with st.expander("📝 Export pre-registration manifest (recommended)"):
        st.caption(
            "Lock in your design before you collect data. Download this JSON and upload it on "
            "the **Analyze Results** page — the recommendation will be stamped with a hash so "
            "reviewers can verify the analysis matches the registered plan."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            _exp_id = st.text_input("Experiment ID / name", value="my-experiment")
            _primary = st.text_input("Primary metric", value="conversion_rate")
        with col_b:
            _hypothesis = st.text_area(
                "Hypothesis (one sentence)",
                value="Treatment increases the primary metric vs control.",
                height=80,
            )
        # Decision rule is fixed — it documents the dual-evidence rule baked
        # into this toolkit's engine.  Most industry teams use ONE school
        # (frequentist OR Bayesian), not both — see the per-school decision
        # matrices in the README for a more standard setup.
        _decision_rule = (
            "Ship if frequentist p < α AND Bayesian P(B>A) ≥ 95% "
            "AND no SRM AND no significant Simpson reversal AND not Twyman-flagged."
        )
        st.markdown(
            f"""<div style="background:#f4f7ff;border:1px solid #cdd9f5;border-radius:8px;
                padding:0.7rem 1rem;margin:0.4rem 0 0.6rem 0;">
                <div style="font-size:0.78rem;color:#5a6a8a;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;margin-bottom:0.3rem;">
                    Decision rule (fixed)
                </div>
                <div style="font-family:'Monaco','Menlo',monospace;font-size:0.88rem;color:#0a1f4d;font-weight:600;">
                    {_decision_rule}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "⚠️ **Heads-up:** dual-evidence (frequentist *and* Bayesian) is **not the "
            "industry default**. Most teams pick one school: pure frequentist "
            "(p-value + CI, e.g., classic Microsoft/Booking ExP setups) or pure "
            "Bayesian (P(B>A) + expected loss, e.g., Convoy, VWO). This toolkit "
            "requires both as a teaching device — it forces you to inspect both "
            "lenses and surfaces conflicts."
        )

        st.markdown("---")
        st.markdown("**Lock policy knobs into the manifest**")
        st.caption(
            "Optional: snapshot the structured decision-engine policy into the "
            "manifest so drift between the registered plan and the as-run analysis "
            "is automatically detected on the **Analyze Results** page."
        )
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            _manifest_loss_tol = st.number_input(
                "loss_tolerance", min_value=0.0,
                value=float(st.session_state.get("loss_tolerance", 0.0) or 0.0),
                step=0.0005, format="%.5f",
                help=(
                    "**What:** maximum Bayesian expected loss (in absolute metric units) "
                    "you're willing to tolerate before shipping.\n\n"
                    "**Impact:** when the engine would otherwise say 'Ship', it gets "
                    "downgraded to 'Inconclusive' if Bayesian expected loss exceeds "
                    "this value. Acts as an asymmetric brake — stops you from shipping "
                    "when the downside is too large, even if the average looks positive.\n\n"
                    "**Example:** for a conversion-rate experiment, `0.001` = tolerate "
                    "at most a 0.1pp drop in expectation. **Set to 0 to disable.**"
                ),
            )
            _manifest_practical = st.number_input(
                "practical_significance_threshold", min_value=0.0,
                value=float(st.session_state.get("practical_significance_threshold", 0.0) or 0.0),
                step=0.001, format="%.4f",
                help=(
                    "**What:** smallest effect size your business actually cares about "
                    "(absolute units).\n\n"
                    "**Impact:** when a non-significant result has its CI fully inside "
                    "±this value, the engine reports **'No Effect'** instead of "
                    "**'Inconclusive'** — i.e., you've ruled out a meaningful effect, "
                    "not just failed to detect one.\n\n"
                    "**Example:** `0.005` = 0.5pp lift on a conversion metric. "
                    "**Set to 0 to disable** — non-significant results stay 'Inconclusive'."
                ),
            )
        with col_p2:
            _manifest_swm = st.checkbox(
                "allow_ship_with_monitoring",
                value=bool(st.session_state.get("allow_ship_with_monitoring", False)),
                help=(
                    "**What:** enable the intermediate **'Ship with Monitoring'** state "
                    "between 'Inconclusive' and 'Ship'.\n\n"
                    "**Impact when ON:** when the frequentist test is inconclusive but "
                    "Bayesian P(B>A) ≥ the monitoring threshold (default 85%) and "
                    "expected loss is small, the engine recommends **shipping behind a "
                    "small holdback** rather than running longer. Useful for low-risk "
                    "changes where extending the test is costly.\n\n"
                    "**When OFF:** the engine never returns 'Ship with Monitoring' — "
                    "you only get Ship / No-Ship / Inconclusive / No Effect."
                ),
            )
            _manifest_mpt = st.number_input(
                "monitoring_prob_threshold", min_value=0.5, max_value=0.99,
                value=float(st.session_state.get("monitoring_prob_threshold", 0.85)),
                step=0.01, format="%.2f",
                help=(
                    "**What:** Bayesian P(B>A) cutoff that triggers 'Ship with Monitoring' "
                    "(only relevant when `allow_ship_with_monitoring` is ON).\n\n"
                    "**Impact:** higher values = stricter — fewer SwM recommendations, "
                    "more 'Inconclusive'. Lower values = more permissive — more SwM "
                    "recommendations.\n\n"
                    "**Default:** `0.85` (a moderate evidence bar). Range: 0.50 – 0.99."
                ),
            )
        with col_p3:
            _manifest_twy = st.number_input(
                "twyman_min_baseline", min_value=0.0001,
                value=float(st.session_state.get("twyman_min_baseline", 0.01)),
                step=0.001, format="%.4f",
                help=(
                    "**What:** baseline-rate floor (for proportion metrics) below which "
                    "Twyman's Law sanity-checking engages.\n\n"
                    "**Twyman's Law:** *'any figure that looks interesting is probably "
                    "wrong.'* If you observe a 200% lift on a 0.001% metric, it's almost "
                    "always a tracking bug, not a real effect. The engine flags such "
                    "results.\n\n"
                    "**Impact:** lowering this catches more low-baseline experiments in "
                    "the Twyman net (more flags, more friction). Default `0.01` = 1%; "
                    "metrics below 1% baseline get extra scrutiny."
                ),
            )
            _manifest_lift = st.number_input(
                "lift_warning_threshold", min_value=0.0,
                value=float(st.session_state.get("lift_warning_threshold", 0.50)),
                step=0.05, format="%.2f",
                help=(
                    "**What:** relative-lift size (e.g., `0.50` = 50%) above which the "
                    "engine emits a 'too-good-to-be-true' warning.\n\n"
                    "**Impact:** when treatment lifts the metric by more than this "
                    "fraction relative to control, a warning flag is added prompting "
                    "you to verify instrumentation, randomization, and sample composition "
                    "before celebrating. Real product changes rarely produce >50% lift "
                    "on top-line metrics.\n\n"
                    "**Default:** `0.50` (50% lift). Set higher to mute, lower to be "
                    "more paranoid."
                ),
            )
        from ab_test_toolkit import __version__ as _toolkit_version
        _manifest = {
            "manifest_version": 2,
            "toolkit_version": _toolkit_version,
            "experiment_id": _exp_id,
            "primary_metric": _primary,
            "hypothesis": _hypothesis,
            "decision_rule": _decision_rule,
            "alpha": float(alpha),
            "power": float(power),
            "metric_type": "proportion" if is_proportion else "continuous",
            "mde": float(mde),
            "mde_mode": mde_mode,
            "allocation_ratio": float(allocation_ratio),
            "planned_n_control": int(result.n_control),
            "planned_n_treatment": int(result.n_treatment),
            "planned_n_total": int(result.n_total),
            "estimated_days": result.estimated_days,
            # Decision-engine policy lock-in (verified on Analyze)
            "loss_tolerance": float(_manifest_loss_tol) if _manifest_loss_tol > 0 else None,
            "allow_ship_with_monitoring": bool(_manifest_swm),
            "monitoring_prob_threshold": float(_manifest_mpt),
            "twyman_min_baseline": float(_manifest_twy),
            "practical_significance_threshold": float(_manifest_practical) if _manifest_practical > 0 else None,
            "lift_warning_threshold": float(_manifest_lift),
            # Bayesian seeds (for reproducibility)
            "bayesian_random_state": 0,
            "bayesian_n_simulations": 100_000,
        }
        if is_proportion:
            _manifest["baseline_rate"] = float(baseline_rate)
        else:
            _manifest["baseline_mean"] = float(baseline_mean)
            _manifest["baseline_std"] = float(baseline_std)
        import json as _json
        st.download_button(
            "⬇️ Download manifest.json",
            data=_json.dumps(_manifest, indent=2).encode("utf-8"),
            file_name=f"{_exp_id}_manifest.json",
            mime="application/json",
        )

    # --- Charts ---
    st.markdown("")
    if is_proportion:
        section_header("MDE vs. Sample Size", "How required sample size changes with effect size", "📈")
        # Convert to absolute MDE for chart if user selected relative mode
        abs_mde = mde * baseline_rate if mde_mode == "relative" else mde
        # Cap upper MDE to keep p1 < 1
        max_valid_mde = (1.0 - baseline_rate) * 0.95
        mde_upper = min(max_valid_mde, max(abs_mde * 4, 0.20))
        mde_lower = max(0.001, abs_mde * 0.25)
        pc_data = power_curve(
            baseline_rate=baseline_rate,
            mde_range=(mde_lower, mde_upper),
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
        )
        fig_mde = mde_vs_n_curve(pc_data)
        st.plotly_chart(fig_mde, use_container_width=True)

        st.markdown("")
        section_header("Sample-Size Inflation vs. Allocation Ratio", "Extra sample required when the split is unbalanced (target MDE and power held fixed)", "⚖️")
        fig_pl = power_loss_curve(baseline_rate=baseline_rate, mde=abs_mde, alpha=alpha)
        st.plotly_chart(fig_pl, use_container_width=True)
    else:
        info_callout(
            "Power and allocation charts above are proportion-specific. For continuous metrics, "
            "use the **sample-size table** at the top of this page — it is fully supported and "
            "uses the Welch-aware formula. The visual curves for continuous metrics are not "
            "yet implemented.",
            "info",
        )

except Exception as exc:
    display_error(f"Could not compute sample sizes: {exc}")
