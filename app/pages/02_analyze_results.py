"""📊 Analyze Results — upload CSV or enter summary stats for full analysis."""

import numpy as np
import pandas as pd
import streamlit as st

from ab_test_toolkit.bayesian import beta_binomial, beta_binomial_from_stats, normal_normal, normal_normal_from_stats
from ab_test_toolkit.cuped import cuped_adjust
from ab_test_toolkit.frequentist import welch_t_test, welch_t_test_from_stats, z_test, z_test_from_stats
from ab_test_toolkit.io import load_experiment_data
from ab_test_toolkit.recommendation import generate_recommendation, check_novelty
from ab_test_toolkit.segmentation import segment_analysis
from ab_test_toolkit.srm import check_srm, check_srm_by_stratum
from ab_test_toolkit.visualization import (
    ci_comparison_plot,
    posterior_plot,
    segment_comparison_chart,
)

from app_utils import (
    display_error,
    get_allow_ship_with_monitoring,
    get_alpha,
    get_loss_tolerance,
    get_monitoring_prob_threshold,
    get_practical_significance_threshold,
    get_uploaded_manifest,
    info_callout,
    metric_row,
    page_header,
    render_header_credit,
    render_sidebar_settings,
    section_header,
    status_badge,
)

render_header_credit()
render_sidebar_settings()


def _render_recommendation(rec):
    """Render a Recommendation object with consistent styling."""
    color_map = {
        "Ship": "green", "Ship with Monitoring": "blue",
        "No-Ship": "red", "Inconclusive": "orange", "No Effect": "gray",
    }
    bg_map = {
        "Ship": "#edfaf3", "Ship with Monitoring": "#eef4ff",
        "No-Ship": "#fef0f0", "Inconclusive": "#fff8ed", "No Effect": "#f4f5f7",
    }
    border_map = {
        "Ship": "#0a8754", "Ship with Monitoring": "#1f63d6",
        "No-Ship": "#d32f2f", "Inconclusive": "#e67e22", "No Effect": "#777",
    }
    emoji_map = {
        "Ship": "🚀", "Ship with Monitoring": "🛰️",
        "No-Ship": "🛑", "Inconclusive": "🤔",
    }
    rec_bg = bg_map.get(rec.recommendation, "#f8f9fb")
    rec_border = border_map.get(rec.recommendation, "#ccc")
    rec_emoji = emoji_map.get(rec.recommendation, "")

    st.markdown(
        f"""<div style="background:{rec_bg};border:2px solid {rec_border};border-radius:12px;
            padding:1.5rem 2rem;text-align:center;margin:0.5rem 0 1rem 0;">
            <span style="font-size:2.5rem;">{rec_emoji}</span>
            <h2 style="margin:0.3rem 0 0 0;color:{rec_border};">{rec.recommendation}</h2>
        </div>""",
        unsafe_allow_html=True,
    )

    if rec.reason:
        signal_colors = {"strong": "#0a8754", "moderate": "#e67e22", "weak": "#b0b0b0", "none": "#d32f2f"}
        signal_labels = {"strong": "Strong Signal", "moderate": "Moderate Signal", "weak": "Weak Signal", "none": "No Signal"}
        sig_color = signal_colors.get(rec.signal_strength, "#555")
        sig_label = signal_labels.get(rec.signal_strength, "")
        signal_badge = (
            f'<span style="background:{sig_color}15;color:{sig_color};padding:0.2rem 0.6rem;'
            f'border-radius:12px;font-size:0.75rem;font-weight:700;margin-left:0.5rem;">'
            f'{sig_label}</span>'
            if sig_label else ""
        )
        st.markdown(
            f"""<div style="background:#f8f9fb;border:1px solid #e2e6ed;border-radius:10px;
                padding:1rem 1.3rem;margin:0.5rem 0 1rem 0;font-size:0.92rem;color:#444;line-height:1.5;">
                <strong>Why this decision</strong>{signal_badge}<br><br>
                {rec.reason}
            </div>""",
            unsafe_allow_html=True,
        )

    if rec.flags:
        for flag in rec.flags:
            info_callout(flag, "warning")

    if rec.next_steps:
        st.markdown("")
        st.markdown(
            """<div style="background:#f0f4ff;border:1px solid #b3c7f7;border-radius:10px;
                padding:1.2rem 1.5rem;margin:0.5rem 0 1rem 0;">
                <span style="font-size:1.1rem;font-weight:700;color:#1A1A2E;">💡 Suggested Next Steps</span>
            </div>""",
            unsafe_allow_html=True,
        )
        for i, step in enumerate(rec.next_steps, 1):
            st.markdown(
                f"""<div style="background:#fafbfc;border-left:3px solid #0066FF;border-radius:0 6px 6px 0;
                    padding:0.7rem 1rem;margin:0.4rem 0;font-size:0.92rem;color:#333;">
                    <strong>{i}.</strong> {step}
                </div>""",
                unsafe_allow_html=True,
            )

    if getattr(rec, "manifest_drift", None):
        _drift_html = "".join(f"<li>{d}</li>" for d in rec.manifest_drift)
        st.markdown(
            f"""<div style="background:#fff4f4;border:2px solid #d32f2f;border-radius:10px;
                padding:1rem 1.3rem;margin:0.5rem 0 1rem 0;">
                <strong style="color:#d32f2f;">⚠️ Manifest drift detected</strong><br>
                <span style="font-size:0.88rem;color:#444;">The as-run analysis differs from the registered plan on:</span>
                <ul style="margin:0.4rem 0 0 1.2rem;font-size:0.85rem;color:#333;">{_drift_html}</ul>
                <span style="font-size:0.8rem;color:#666;">The recommendation reflects as-run settings; treat it as exploratory unless drift is justified.</span>
            </div>""",
            unsafe_allow_html=True,
        )

    if getattr(rec, "manifest_hash", None):
        _full_hash = getattr(rec, "manifest_hash_full", None) or rec.manifest_hash
        st.markdown(
            f"""<div style="background:#f4f7ff;border:1px solid #cdd9f5;border-radius:8px;
                padding:0.6rem 1rem;margin:0.4rem 0;font-size:0.85rem;color:#1f3a7a;">
                🔖 <strong>Pre-reg manifest stamped</strong> · short hash <code>{rec.manifest_hash}</code> ·
                <span style="font-size:0.75rem;color:#5a6a8a;">full SHA-256 in supporting metrics</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("📋 Supporting metrics"):
        sm = dict(rec.supporting_metrics)
        if getattr(rec, "manifest_hash_full", None):
            sm["manifest_hash_sha256"] = rec.manifest_hash_full
        st.json(sm)


page_header("📊", "Analyze Results", "Upload CSV or enter summary stats for full frequentist & Bayesian analysis.")

info_callout(
    "**Trustworthy analysis is pre-registered analysis.** Before uploading data, ideally fix "
    "(in writing) your primary metric, MDE, sample size, decision threshold, and analysis plan. "
    "This toolkit cannot enforce pre-registration — running multiple specifications until one "
    "looks favourable inflates false-positive rates and invalidates the reported confidence levels.",
    "info",
)

# ── Data input toggle ────────────────────────────────────────────────────────
st.markdown(
    """<div style="background:#f8f9fb;border:1px solid #e2e6ed;border-radius:10px;
        padding:0.8rem 1.2rem;margin-bottom:1rem;">
        <span style="font-size:0.85rem;color:#777;text-transform:uppercase;letter-spacing:0.5px;">
            📁 Data Input Method
        </span>
    </div>""",
    unsafe_allow_html=True,
)
input_mode = st.radio("Data input method", ["Upload CSV", "Manual summary stats"], horizontal=True, label_visibility="collapsed")

alpha = get_alpha()
df: pd.DataFrame | None = None
is_proportion = True
has_covariate = False
has_segment = False

# ── Decision-engine advanced settings ──────────────────────────────────────
with st.expander("⚙️ Advanced decision settings (optional)"):
    st.caption(
        "Tighten or relax the recommendation engine. Defaults match the documented "
        "decision matrices; change these only with stakeholder alignment, ideally "
        "captured in a pre-reg manifest below."
    )

    # Process manifest upload FIRST so defaults can be seeded from it.
    st.markdown("**Pre-registration manifest**")
    st.caption(
        "Upload the JSON manifest you exported from the Experiment Design page. "
        "Registered policy fields will be auto-loaded into the controls below, "
        "and any drift between the registered plan and the as-run analysis will "
        "be flagged on the recommendation."
    )
    _manifest_upload = st.file_uploader("Pre-reg manifest (JSON)", type=["json"], key="manifest_upload")
    if _manifest_upload is not None:
        try:
            import json as _json
            _loaded = _json.loads(_manifest_upload.read())
            st.session_state["uploaded_manifest"] = _loaded
            # Auto-load policy fields from the manifest into session state so
            # the controls below pick them up as defaults.
            for _k in (
                "loss_tolerance",
                "allow_ship_with_monitoring",
                "monitoring_prob_threshold",
                "twyman_min_baseline",
                "practical_significance_threshold",
                "lift_warning_threshold",
                "alpha",
            ):
                if _k in _loaded and _loaded[_k] is not None:
                    st.session_state[_k] = _loaded[_k]
            st.success(
                f"Manifest loaded ({_loaded.get('experiment_id', 'unnamed')}). "
                f"Policy fields applied to the controls below; drift will be flagged on the result."
            )
        except Exception as _e:
            st.error(f"Could not parse manifest JSON: {_e}")
            st.session_state.pop("uploaded_manifest", None)
    elif "uploaded_manifest" in st.session_state and st.button("Clear loaded manifest"):
        st.session_state.pop("uploaded_manifest", None)

    if "uploaded_manifest" in st.session_state:
        _m = st.session_state["uploaded_manifest"]
        st.info(
            f"📌 Active manifest: **{_m.get('experiment_id', 'unnamed')}** · "
            f"primary={_m.get('primary_metric','?')} · α={_m.get('alpha','?')} · "
            f"planned N={_m.get('planned_n_total','?')} · "
            f"toolkit v{_m.get('toolkit_version','?')}"
        )

    col_lt, col_swm = st.columns(2)
    with col_lt:
        _lt = st.number_input(
            "Expected-loss tolerance (Bayesian, absolute units)",
            min_value=0.0, value=float(st.session_state.get("loss_tolerance", 0.0) or 0.0),
            step=0.0005, format="%.5f",
            help="A 'Ship' is downgraded to Inconclusive when Bayesian expected loss exceeds this. "
                 "Leave 0 to disable.",
        )
        st.session_state["loss_tolerance"] = _lt
    with col_swm:
        _swm = st.checkbox(
            "Allow 'Ship with Monitoring' (intermediate state)",
            value=st.session_state.get("allow_ship_with_monitoring", False),
            help="When the frequentist test is inconclusive but Bayesian P(B>A) is above the "
                 "monitoring threshold and expected loss is small, recommend shipping behind a "
                 "small holdback rather than running longer.",
        )
        st.session_state["allow_ship_with_monitoring"] = _swm
        _mpt = st.slider(
            "Monitoring P(B>A) threshold",
            min_value=0.70, max_value=0.94,
            value=float(st.session_state.get("monitoring_prob_threshold", 0.85)),
            step=0.01, disabled=not _swm,
        )
        st.session_state["monitoring_prob_threshold"] = _mpt

# ── Data ingestion ──────────────────────────────────────────────────────────
if input_mode == "Upload CSV":
    # Clear manual state to prevent stale results from overriding CSV analysis
    st.session_state.pop("_manual_proportion", None)
    st.session_state.pop("_manual_continuous", None)
    info_callout(
        "Your CSV should have a **group** column (with values `control` and `treatment`) "
        "and a **value** column (0/1 for conversion data, or continuous for revenue/time metrics). "
        "Optional columns: **segment** (e.g., mobile/desktop), **covariate** (pre-experiment metric value for CUPED — must be measured BEFORE the experiment started), "
        "and **day** (integer day number from experiment start, for novelty effect detection).",
        callout_type="info",
    )
    with st.expander("💡 What is a novelty effect and how is it detected?"):
        st.markdown(
            "A **novelty effect** occurs when users react to a change simply because it's new, "
            "inflating early results that fade over time.\n\n"
            "**How to detect it:** Include a `day` column in your CSV (integer day number from "
            "experiment start). The toolkit will automatically:\n\n"
            "1. Compare the treatment effect in the first ~20% of days vs. the later period\n"
            "2. Heuristically flag a warning if the early lift is at least 2× the stabilized effect\n\n"
            "**This is a heuristic, not a statistical test.** It can produce false positives "
            "(especially with low daily traffic) and false negatives. Treat the flag as a prompt "
            "to investigate, not a verdict.\n\n"
            "**If novelty is detected**, consider re-running the analysis after excluding the "
            "first 3–7 days (burn-in period) to see the longer-term effect. "
            "The Case Study notebook demonstrates this workflow end-to-end."
        )
    info_callout(
        "🔒 **Privacy notice**: this is a personal/educational toolkit, not a hosted enterprise "
        "service. Do not upload PII (names, emails, IPs, raw account IDs) or data subject to "
        "contractual or regulatory restrictions. Use de-identified samples or hashed unit IDs. "
        "Files are read into your Streamlit session's memory only; the toolkit does not write "
        "uploads to disk, but you are responsible for ensuring your environment (browser, "
        "Streamlit Cloud, logs) is acceptable for the data you choose to upload.",
        "info",
    )
    uploaded = st.file_uploader("Upload experiment CSV", type=["csv"])
    if uploaded is not None:
        try:
            # File size cap: 50 MB
            _MAX_BYTES = 50 * 1024 * 1024
            if hasattr(uploaded, "size") and uploaded.size and uploaded.size > _MAX_BYTES:
                raise ValueError(
                    f"File too large ({uploaded.size / 1024 / 1024:.1f} MB). "
                    f"Maximum supported size is {_MAX_BYTES // 1024 // 1024} MB. "
                    f"For larger experiments, pre-aggregate to summary statistics."
                )
            raw_df = pd.read_csv(uploaded)
            # Row cap: 5M rows (defensive — anything larger should pre-aggregate)
            _MAX_ROWS = 5_000_000
            if len(raw_df) > _MAX_ROWS:
                raise ValueError(
                    f"Too many rows ({len(raw_df):,}). Maximum supported is {_MAX_ROWS:,}. "
                    f"Pre-aggregate to summary statistics for very large experiments."
                )
            df, metric_type = load_experiment_data(raw_df)
            is_proportion = metric_type.value == "proportion"
            has_covariate = "covariate" in df.columns
            has_segment = "segment" in df.columns

            # Unit-of-analysis duplicate detection
            if "unit_id" in df.columns:
                _dups = df["unit_id"].duplicated().sum()
                if _dups > 0:
                    st.warning(
                        f"⚠️ **Unit-of-analysis violation risk**: {_dups:,} duplicate `unit_id` "
                        f"values found ({_dups / len(df):.1%} of rows). Most A/B tests assume "
                        f"one row per user — duplicates inflate effective N and shrink CIs. "
                        f"Aggregate to one row per unit (sum/mean as appropriate) before analysis."
                    )

            st.success(
                f"Loaded **{len(df):,}** rows — detected metric type: **{metric_type.value}**"
            )
            if is_proportion:
                st.caption("📊 Metric detected as **proportion** (binary 0/1 values) → Z-test will be used. "
                           "If your data is continuous, ensure the `value` column contains non-binary values.")
            else:
                st.caption("📊 Metric detected as **continuous** (numeric values) → Welch's t-test will be used. "
                           "If your data is binary (0/1), check for unexpected values in the `value` column.")
                # Warn about heavy-tailed data (rewritten in plain English)
                from scipy.stats import kurtosis as _kurtosis
                _kurt = _kurtosis(df["value"].dropna(), fisher=True)
                if _kurt > 10:
                    st.warning(
                        f"⚠️ **A few extreme values are dominating this metric** "
                        f"(excess kurtosis = {_kurt:.1f}; values above ~3 indicate heavy tails). "
                        f"That means a small number of outlier observations carry most of the "
                        f"variance, which makes the test sensitive to chance and the CI wide. "
                        f"Consider winsorizing the metric at the 99th percentile or analysing on "
                        f"a log scale before drawing conclusions."
                    )
        except Exception as exc:
            display_error(str(exc))
else:
    section_header("Enter Summary Statistics", "Enter your experiment data", "✏️")
    info_callout(
        "**Control** = the original experience (Group A, no change).  \n"
        "**Treatment** = the new variation being tested (Group B, the change you're evaluating).",
        callout_type="info",
    )

    manual_metric = st.radio(
        "Metric type",
        ["Proportion (conversions / total)", "Continuous (mean / std / n)"],
        horizontal=True,
        help="Proportion: binary outcomes (converted yes/no). Continuous: numeric outcomes (revenue per user, session time, etc.).",
    )
    manual_is_proportion = manual_metric.startswith("Proportion")

    c1, c2 = st.columns(2, gap="large")
    if manual_is_proportion:
        with c1:
            st.markdown("**🅰️ Control Group** *(original / no change)*")
            control_count = st.number_input("Control conversions", min_value=0, value=500, help="Number of users in the control group who converted")
            control_total = st.number_input("Control total", min_value=1, value=5000, help="Total number of users in the control group")
        with c2:
            st.markdown("**🅱️ Treatment Group** *(new variation)*")
            treatment_count = st.number_input("Treatment conversions", min_value=0, value=550, help="Number of users in the treatment group who converted")
            treatment_total = st.number_input("Treatment total", min_value=1, value=5000, help="Total number of users in the treatment group")
    else:
        with c1:
            st.markdown("**🅰️ Control Group** *(original / no change)*")
            control_mean = st.number_input("Control mean", value=50.0, step=1.0, format="%.2f", help="Average metric value for the control group (e.g., revenue per user)")
            control_std = st.number_input("Control std dev", min_value=0.01, value=15.0, step=0.5, format="%.2f", help="Standard deviation of the control group metric")
            control_n = st.number_input("Control sample size", min_value=2, value=5000, step=100, help="Number of users in the control group")
        with c2:
            st.markdown("**🅱️ Treatment Group** *(new variation)*")
            treatment_mean = st.number_input("Treatment mean", value=52.0, step=1.0, format="%.2f", help="Average metric value for the treatment group")
            treatment_std = st.number_input("Treatment std dev", min_value=0.01, value=15.0, step=0.5, format="%.2f", help="Standard deviation of the treatment group metric")
            treatment_n = st.number_input("Treatment sample size", min_value=2, value=5000, step=100, help="Number of users in the treatment group")

    manual_alloc_pct = st.slider(
        "Planned allocation (% to control)",
        min_value=10,
        max_value=90,
        value=50,
        help="Expected traffic split. Default 50/50. Used for SRM check.",
        key="manual_alloc_pct",
    )
    _manual_expected_ratio = (manual_alloc_pct / 100, 1 - manual_alloc_pct / 100)

    if st.button("🚀 Run Analysis", use_container_width=True):
        if manual_is_proportion:
            # Validate counts don't exceed totals
            if control_count > control_total:
                st.error("Control conversions cannot exceed control total.")
                st.stop()
            if treatment_count > treatment_total:
                st.error("Treatment conversions cannot exceed treatment total.")
                st.stop()
            # Use summary-stat functions directly — no array materialization
            st.session_state["_manual_proportion"] = {
                "control_count": control_count,
                "control_total": control_total,
                "treatment_count": treatment_count,
                "treatment_total": treatment_total,
            }
            is_proportion = True
        else:
            # Use summary-stat functions directly — no synthetic data
            st.session_state["_manual_continuous"] = {
                "control_mean": control_mean,
                "control_std": control_std,
                "control_n": control_n,
                "treatment_mean": treatment_mean,
                "treatment_std": treatment_std,
                "treatment_n": treatment_n,
            }
            is_proportion = False

# ── Analysis pipeline — summary stats path (manual input) ───────────────────
_manual_prop = st.session_state.get("_manual_proportion")
_manual_cont = st.session_state.get("_manual_continuous")

if _manual_prop is not None or _manual_cont is not None:
    if _manual_prop is not None:
        _cc, _ct = _manual_prop["control_count"], _manual_prop["control_total"]
        _tc, _tt = _manual_prop["treatment_count"], _manual_prop["treatment_total"]
        _control_rate = _cc / _ct

        # Progress tracker
        step_names = ["SRM Check", "Frequentist", "Bayesian", "Recommendation"]
        steps_html = "".join(
            f'<span style="background:#0066FF;color:#fff;border-radius:50%;width:24px;height:24px;'
            f'display:inline-flex;align-items:center;justify-content:center;font-size:0.7rem;'
            f'font-weight:700;margin-right:0.3rem;">{i+1}</span>'
            f'<span style="margin-right:1rem;font-size:0.85rem;color:#555;">{name}</span>'
            for i, name in enumerate(step_names)
        )
        st.markdown(
            f'<div style="background:#f8f9fb;border:1px solid #e2e6ed;border-radius:10px;'
            f'padding:0.8rem 1.2rem;margin:1rem 0;display:flex;align-items:center;flex-wrap:wrap;">'
            f'{steps_html}</div>',
            unsafe_allow_html=True,
        )

        # SRM
        section_header("Sample Ratio Mismatch (SRM)", "Step 1 — Verify traffic was split correctly", "1️⃣")
        srm_result = check_srm(observed=(_ct, _tt), expected_ratio=_manual_expected_ratio)
        if srm_result.has_mismatch:
            info_callout(f"SRM detected — χ² = {srm_result.chi2_statistic:.2f}, p = {srm_result.p_value:.4f}.", "warning")
        else:
            info_callout(f"No SRM — χ² = {srm_result.chi2_statistic:.2f}, p = {srm_result.p_value:.4f}. Traffic split looks clean.", "success")

        # Frequentist
        section_header("Frequentist Analysis", "Step 2 — Hypothesis test", "2️⃣")
        freq_result = z_test_from_stats(_cc, _ct, _tc, _tt, alpha=alpha)
        sig_badge = status_badge("✅ Significant", "green") if freq_result.is_significant else status_badge("❌ Not Significant", "red")
        st.markdown(f"**Test**: z_test &nbsp;|&nbsp; **p-value**: {freq_result.p_value:.4f} &nbsp;|&nbsp; {sig_badge}", unsafe_allow_html=True)
        _rel_lift_prop = f" ({freq_result.point_estimate / _control_rate:+.2%} relative)" if _control_rate != 0 else ""
        st.markdown(f"**Point estimate**: {freq_result.point_estimate:.4f}{_rel_lift_prop} &nbsp;|&nbsp; **Effect size**: {freq_result.effect_size:.4f} &nbsp;|&nbsp; **CI**: [{freq_result.ci_lower:.4f}, {freq_result.ci_upper:.4f}]")
        st.plotly_chart(ci_comparison_plot(freq_result), use_container_width=True)

        # Bayesian
        section_header("Bayesian Analysis", "Step 3 — Posterior probability", "3️⃣")
        bayes_result = beta_binomial_from_stats(_cc, _ct, _tc, _tt)

    else:  # _manual_cont
        _cm, _cs, _cn = _manual_cont["control_mean"], _manual_cont["control_std"], _manual_cont["control_n"]
        _tm, _ts, _tn = _manual_cont["treatment_mean"], _manual_cont["treatment_std"], _manual_cont["treatment_n"]
        _control_rate = _cm

        step_names = ["SRM Check", "Frequentist", "Bayesian", "Recommendation"]
        steps_html = "".join(
            f'<span style="background:#0066FF;color:#fff;border-radius:50%;width:24px;height:24px;'
            f'display:inline-flex;align-items:center;justify-content:center;font-size:0.7rem;'
            f'font-weight:700;margin-right:0.3rem;">{i+1}</span>'
            f'<span style="margin-right:1rem;font-size:0.85rem;color:#555;">{name}</span>'
            for i, name in enumerate(step_names)
        )
        st.markdown(
            f'<div style="background:#f8f9fb;border:1px solid #e2e6ed;border-radius:10px;'
            f'padding:0.8rem 1.2rem;margin:1rem 0;display:flex;align-items:center;flex-wrap:wrap;">'
            f'{steps_html}</div>',
            unsafe_allow_html=True,
        )

        # SRM
        section_header("Sample Ratio Mismatch (SRM)", "Step 1 — Verify traffic was split correctly", "1️⃣")
        srm_result = check_srm(observed=(_cn, _tn), expected_ratio=_manual_expected_ratio)
        if srm_result.has_mismatch:
            info_callout(f"SRM detected — χ² = {srm_result.chi2_statistic:.2f}, p = {srm_result.p_value:.4f}.", "warning")
        else:
            info_callout(f"No SRM — χ² = {srm_result.chi2_statistic:.2f}, p = {srm_result.p_value:.4f}. Traffic split looks clean.", "success")

        # Frequentist
        section_header("Frequentist Analysis", "Step 2 — Hypothesis test", "2️⃣")
        freq_result = welch_t_test_from_stats(_cm, _cs, _cn, _tm, _ts, _tn, alpha=alpha)
        sig_badge = status_badge("✅ Significant", "green") if freq_result.is_significant else status_badge("❌ Not Significant", "red")
        st.markdown(f"**Test**: welch_t_test &nbsp;|&nbsp; **p-value**: {freq_result.p_value:.4f} &nbsp;|&nbsp; {sig_badge}", unsafe_allow_html=True)
        _rel_lift = f" ({freq_result.point_estimate / _cm:+.2%} relative)" if _cm != 0 else ""
        st.markdown(f"**Point estimate**: {freq_result.point_estimate:.4f}{_rel_lift} &nbsp;|&nbsp; **Effect size**: {freq_result.effect_size:.4f} &nbsp;|&nbsp; **CI**: [{freq_result.ci_lower:.4f}, {freq_result.ci_upper:.4f}]")
        st.plotly_chart(ci_comparison_plot(freq_result), use_container_width=True)

        # Bayesian
        section_header("Bayesian Analysis", "Step 3 — Posterior probability", "3️⃣")
        bayes_result = normal_normal_from_stats(_cm, _cs, _cn, _tm, _ts, _tn)

    # Common rendering for manual stats path
    metric_row([
        {
            "icon": "🎲", "label": "P(Treatment > Control)", "value": f"{bayes_result.prob_b_greater_a:.2%}",
            "help": "Posterior probability — answers: 'Given the data, what is the probability Treatment is truly better than Control?' Unlike a p-value (which assumes no difference and asks how surprising the data is), this directly tells you the likelihood B beats A. Above 95% is typically strong evidence.",
        },
        {
            "icon": "📉", "label": "Expected Loss", "value": f"{bayes_result.expected_loss:.5f}",
            "help": "The expected loss (regret) if you choose treatment and it turns out to be worse than control. Lower is better — a value close to 0 means little downside risk to shipping the treatment.",
        },
        {
            "icon": "📐", "label": "95% Credible Interval",
            "value": f"[{bayes_result.credible_interval[0]:.4f}, {bayes_result.credible_interval[1]:.4f}]",
            "help": "The Bayesian 95% credible interval for the treatment effect (lift). There is a 95% posterior probability that the true effect falls within this range.",
        },
    ])
    st.markdown("")
    st.plotly_chart(posterior_plot(bayes_result), use_container_width=True)

    # Recommendation
    st.markdown("")
    section_header("Recommendation", "Final verdict based on all evidence", "🏁")
    rec = generate_recommendation(
        frequentist=freq_result,
        bayesian=bayes_result,
        srm=srm_result,
        control_rate=_control_rate,
        practical_significance_threshold=get_practical_significance_threshold(),
        loss_tolerance=get_loss_tolerance(),
        allow_ship_with_monitoring=get_allow_ship_with_monitoring(),
        monitoring_prob_threshold=get_monitoring_prob_threshold(),
        manifest=get_uploaded_manifest(),
    )
    _render_recommendation(rec)

elif df is not None:
    # Validate group labels
    groups = set(df["group"].unique())
    if "control" not in groups or "treatment" not in groups:
        st.error(
            f"CSV `group` column must contain exactly **'control'** and **'treatment'**. "
            f"Found: {sorted(groups)}. Please rename your group labels."
        )
        st.stop()

    control = df.loc[df["group"] == "control", "value"].to_numpy()
    treatment = df.loc[df["group"] == "treatment", "value"].to_numpy()

    # Pipeline progress tracker
    total_steps = 4 + (1 if "covariate" in df.columns else 0) + (1 if "segment" in df.columns else 0)
    step_names = ["SRM Check", "Frequentist", "Bayesian"]
    if "covariate" in df.columns:
        step_names.append("CUPED")
    if "segment" in df.columns:
        step_names.append("Segments")
    step_names.append("Recommendation")

    steps_html = "".join(
        f'<span style="background:#0066FF;color:#fff;border-radius:50%;width:24px;height:24px;'
        f'display:inline-flex;align-items:center;justify-content:center;font-size:0.7rem;'
        f'font-weight:700;margin-right:0.3rem;">{i+1}</span>'
        f'<span style="margin-right:1rem;font-size:0.85rem;color:#555;">{name}</span>'
        for i, name in enumerate(step_names)
    )
    st.markdown(
        f'<div style="background:#f8f9fb;border:1px solid #e2e6ed;border-radius:10px;'
        f'padding:0.8rem 1.2rem;margin:1rem 0;display:flex;align-items:center;flex-wrap:wrap;">'
        f'{steps_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # 1. SRM check
    section_header("Sample Ratio Mismatch (SRM)", "Step 1 — Verify traffic was split correctly", "1️⃣")
    srm_ratio_input = st.slider(
        "Planned allocation (% to control)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        help="The intended traffic split. 50 = equal 50/50 split. 80 = 80% control / 20% treatment.",
    )
    srm_expected = (srm_ratio_input / 100, 1 - srm_ratio_input / 100)
    srm_result = check_srm(observed=(len(control), len(treatment)), expected_ratio=srm_expected)
    if srm_result.has_mismatch:
        info_callout(
            f"SRM detected — χ² = {srm_result.chi2_statistic:.2f}, "
            f"p = {srm_result.p_value:.4f}. Interpret results with caution.",
            "warning",
        )
    else:
        info_callout(
            f"No SRM — χ² = {srm_result.chi2_statistic:.2f}, "
            f"p = {srm_result.p_value:.4f}. Traffic split looks clean.",
            "success",
        )

    # 1b. Stratified SRM (per-day) — surfaces day-level bucketing bugs that the
    # aggregate test misses (Holm-corrected across days).
    if "day" in df.columns:
        try:
            stratum_srm = check_srm_by_stratum(
                df, group_col="group", stratum_col="day",
                expected_ratio=srm_expected, threshold=0.01,
            )
            if stratum_srm.n_strata > 1:
                with st.expander(
                    f"📅 Stratified SRM by day "
                    f"({stratum_srm.n_mismatches}/{stratum_srm.n_strata} days flagged after Holm correction)"
                ):
                    if stratum_srm.warning:
                        info_callout(stratum_srm.warning, "warning")
                    else:
                        info_callout(
                            "All daily strata are within the planned allocation after Holm correction.",
                            "success",
                        )
                    if stratum_srm.stratum_results:
                        _strat_df = pd.DataFrame(stratum_srm.stratum_results)[
                            ["stratum", "n_control", "n_treatment", "p_value", "p_value_adjusted", "has_mismatch"]
                        ]
                        st.dataframe(_strat_df, hide_index=True, use_container_width=True)
        except Exception as _e:
            st.caption(f"Stratified SRM skipped: {_e}")

    # 2. Frequentist test
    section_header("Frequentist Analysis", "Step 2 — Hypothesis test with p-value and confidence interval", "2️⃣")
    if is_proportion:
        freq_result = z_test(control, treatment, alpha=alpha)
    else:
        freq_result = welch_t_test(control, treatment, alpha=alpha)

    sig_badge = status_badge("✅ Significant", "green") if freq_result.is_significant else status_badge("❌ Not Significant", "red")
    st.markdown(
        f"**Test**: {freq_result.test_type} &nbsp;|&nbsp; **p-value**: {freq_result.p_value:.4f} &nbsp;|&nbsp; {sig_badge}",
        unsafe_allow_html=True,
    )
    _csv_ctrl_mean = float(control.mean())
    _csv_rel = f" ({freq_result.point_estimate / _csv_ctrl_mean:+.2%} relative)" if _csv_ctrl_mean != 0 else ""
    st.markdown(
        f"**Point estimate**: {freq_result.point_estimate:.4f}{_csv_rel} &nbsp;|&nbsp; "
        f"**Effect size**: {freq_result.effect_size:.4f} &nbsp;|&nbsp; "
        f"**CI**: [{freq_result.ci_lower:.4f}, {freq_result.ci_upper:.4f}]"
    )
    st.plotly_chart(ci_comparison_plot(freq_result), use_container_width=True)

    # 3. Bayesian test
    section_header("Bayesian Analysis", "Step 3 — Posterior probability and expected loss", "3️⃣")
    if is_proportion:
        bayes_result = beta_binomial(control, treatment)
    else:
        bayes_result = normal_normal(control, treatment)

    metric_row([
        {
            "icon": "🎲",
            "label": "P(Treatment > Control)",
            "value": f"{bayes_result.prob_b_greater_a:.2%}",
            "help": "Posterior probability — answers: 'Given the data, what is the probability Treatment is truly better than Control?' Unlike a p-value (which assumes no difference and asks how surprising the data is), this directly tells you the likelihood B beats A. Above 95% is typically strong evidence.",
        },
        {
            "icon": "📉",
            "label": "Expected Loss",
            "value": f"{bayes_result.expected_loss:.5f}",
            "help": "The expected loss (regret) if you choose treatment and it turns out to be worse than control. Lower is better — a value close to 0 means little downside risk to shipping the treatment.",
        },
        {
            "icon": "📐",
            "label": "95% Credible Interval",
            "value": f"[{bayes_result.credible_interval[0]:.4f}, {bayes_result.credible_interval[1]:.4f}]",
            "help": "The Bayesian 95% credible interval for the treatment effect (lift). There is a 95% posterior probability that the true effect falls within this range. Unlike a frequentist CI, this can be interpreted as a direct probability statement.",
        },
    ])
    st.markdown("")
    st.plotly_chart(posterior_plot(bayes_result), use_container_width=True)

    # 4. CUPED (optional)
    cuped_result = None
    if has_covariate:
        section_header("CUPED Variance Reduction", "Step 4 — Pre-experiment covariate adjustment", "4️⃣")
        ctrl_cov = df.loc[df["group"] == "control", "covariate"].to_numpy()
        treat_cov = df.loc[df["group"] == "treatment", "covariate"].to_numpy()

        # Covariate balance check: groups should have similar covariate
        # distributions if randomization succeeded.  Large imbalances bias
        # CUPED.
        from scipy import stats as _sp
        try:
            _bal_t, _bal_p = _sp.ttest_ind(ctrl_cov, treat_cov, equal_var=False)
            _ctrl_mean, _treat_mean = float(ctrl_cov.mean()), float(treat_cov.mean())
            _pooled_sd = float(((ctrl_cov.std(ddof=1) ** 2 + treat_cov.std(ddof=1) ** 2) / 2) ** 0.5)
            _smd = abs(_treat_mean - _ctrl_mean) / _pooled_sd if _pooled_sd > 0 else 0.0
            if _bal_p < 0.01 or _smd > 0.10:
                info_callout(
                    f"⚠️ **Covariate imbalance detected** (control mean={_ctrl_mean:.4f}, "
                    f"treatment mean={_treat_mean:.4f}, SMD={_smd:.3f}, p={_bal_p:.4f}). "
                    f"CUPED assumes the covariate is balanced across arms — large imbalance "
                    f"can bias the adjusted estimate. Investigate randomization before relying "
                    f"on the CUPED-adjusted result.",
                    "warning",
                )
        except Exception:
            pass

        cuped_result = cuped_adjust(control, treatment, ctrl_cov, treat_cov, alpha=alpha)

        # CUPED quality gate: low correlation → CUPED won't help, may hurt
        if abs(cuped_result.correlation) < 0.10:
            info_callout(
                f"⚠️ **Weak covariate** (|corr|={abs(cuped_result.correlation):.2f} < 0.10). "
                f"CUPED needs a covariate correlated with the outcome to reduce variance. "
                f"At this correlation the adjusted estimate is essentially the unadjusted one — "
                f"consider a more predictive pre-experiment metric.",
                "info",
            )

        metric_row([
            {"icon": "📊", "label": "Variance Reduction (theoretical)", "value": f"{cuped_result.variance_reduction_pct:.1f}%",
             "help": "ρ² × 100 — the asymptotic upper bound on variance reduction."},
            {"icon": "🎯", "label": "Variance Reduction (realized)", "value": f"{cuped_result.realized_variance_reduction_pct:.1f}%",
             "help": "Observed reduction based on adjusted vs unadjusted CI widths on this dataset."},
            {"icon": "📏", "label": "Unadjusted Estimate", "value": f"{cuped_result.unadjusted_estimate:.4f}"},
            {"icon": "✨", "label": "CUPED-Adjusted Estimate", "value": f"{cuped_result.adjusted_estimate:.4f}"},
        ])

    # 5. Segmentation (optional)
    seg_result = None
    if has_segment:
        step_num = "5️⃣" if has_covariate else "4️⃣"
        section_header("Segment Analysis", f"Step {'5' if has_covariate else '4'} — Heterogeneous treatment effects (exploratory)", step_num)
        info_callout(
            "Segment-level results are **exploratory**. Per-segment p-values are Holm-adjusted "
            "for the segments you supplied, but the *choice* of segmentation variable is itself "
            "a researcher degree of freedom — running many cuts inflates false-positive risk. "
            "Pre-register segments where possible and treat unexpected wins as hypotheses for "
            "follow-up tests, not as launch decisions.",
            "info",
        )
        seg_result = segment_analysis(df)
        if seg_result.simpsons_paradox:
            info_callout(f"Simpson's Paradox detected — {seg_result.simpsons_details}", "warning")
        for seg in seg_result.segment_results:
            raw_p = seg["p_value"]
            adj_p = seg.get("p_value_adjusted", raw_p)
            if raw_p != raw_p:  # NaN check
                badge = status_badge("insufficient data", "gray")
                st.markdown(f"**{seg['segment']}** — insufficient data (n={seg['n']}) {badge}", unsafe_allow_html=True)
            else:
                sig_color = "green" if adj_p < alpha else "gray"
                badge = status_badge(f"p={raw_p:.4f} (adj={adj_p:.4f})", sig_color)
                st.markdown(
                    f"**{seg['segment']}** — effect: {seg['estimate']:.4f}, "
                    f"CI: [{seg['ci'][0]:.4f}, {seg['ci'][1]:.4f}] {badge}",
                    unsafe_allow_html=True,
                )
        st.plotly_chart(segment_comparison_chart(seg_result), use_container_width=True)

    # 6. Recommendation
    st.markdown("")
    section_header("Recommendation", "Final verdict based on all evidence", "🏁")

    novelty_result = None
    if "day" in df.columns:
        novelty_result = check_novelty(df)
        if novelty_result.has_novelty:
            info_callout(novelty_result.details, "warning")

    # Compute control rate for Twyman relative lift check
    _csv_control_rate = float(control.mean())

    rec = generate_recommendation(
        frequentist=freq_result,
        bayesian=bayes_result,
        srm=srm_result,
        segmentation=seg_result,
        novelty=novelty_result,
        has_covariate=has_covariate,
        control_rate=_csv_control_rate,
        practical_significance_threshold=get_practical_significance_threshold(),
        loss_tolerance=get_loss_tolerance(),
        allow_ship_with_monitoring=get_allow_ship_with_monitoring(),
        monitoring_prob_threshold=get_monitoring_prob_threshold(),
        manifest=get_uploaded_manifest(),
    )
    _render_recommendation(rec)
