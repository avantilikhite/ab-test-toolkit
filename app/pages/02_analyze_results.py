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
    render_diagnostics_card,
    render_header_credit,
    render_sidebar_settings,
    section_header,
    status_badge,
)

render_header_credit()
render_sidebar_settings()


def _srm_check_to_dict(srm_result) -> dict:
    """Convert an SRM result into a Diagnostics-card row dict."""
    chi2 = srm_result.chi2_statistic
    p = srm_result.p_value
    if not srm_result.has_mismatch:
        return {
            "name": "Traffic split (SRM)",
            "status": "pass",
            "summary": f"χ² = {chi2:.2f}, p = {p:.4f}",
        }
    tier = "severe" if p < 1e-6 else "detected" if p < 1e-3 else "borderline"
    return {
        "name": "Traffic split (SRM)",
        "status": "fail",
        "summary": f"χ² = {chi2:.2f}, p = {p:.6f} · {tier}",
        "detail": (
            "Randomisation has broken; no statistical adjustment can recover validity. "
            "Do not interpret these results. Fix the engineering cause (assignment, "
            "attrition pipeline, bot filtering, or a deploy during the window) and re-run."
        ),
    }


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

st.markdown(
    """<details style="background:linear-gradient(135deg,#1A1A2E 0%,#16213e 100%);
        border-radius:12px;padding:0.6rem 1.5rem;margin-top:-1rem;margin-bottom:1.5rem;
        color:#a0b0c8;cursor:pointer;">
        <summary style="color:#fff;font-size:0.95rem;padding:0.4rem 0;font-weight:500;
            list-style:none;outline:none;">
            ✍️ Consider pre-registration
        </summary>
        <p style="margin:0.6rem 0 0.4rem 0;font-size:0.9rem;line-height:1.6;color:#cbd5e1;">
            <strong style="color:#fff;">Trustworthy analysis is pre-registered analysis.</strong>
            Before uploading data, ideally fix (in writing) your primary metric, MDE, sample size,
            decision threshold, and analysis plan. This toolkit cannot enforce pre-registration —
            running multiple specifications until one looks favourable inflates false-positive
            rates and invalidates the reported confidence levels.
        </p>
    </details>""",
    unsafe_allow_html=True,
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

def _render_preregistration_block(key_suffix: str) -> None:
    """Render the manifest uploader and (optional) decision-engine policy knobs.

    Rendered just above each input mode's analysis trigger so the user can
    register a pre-committed plan immediately before the analysis runs.
    `key_suffix` keeps widget keys unique across the two mode call sites.
    """
    # Dynamic expander label: surface manifest status without forcing expansion
    _active_manifest = st.session_state.get("uploaded_manifest")
    if _active_manifest is not None:
        _exp_id = _active_manifest.get("experiment_id", "unnamed")
        _preregistration_label = f"📋 Pre-registration manifest — active: {_exp_id}"
    else:
        _preregistration_label = "📋 Pre-registration manifest (optional)"

    with st.expander(_preregistration_label, expanded=False):
        st.caption(
            "Upload the JSON manifest you exported from the Experiment Design page; "
            "registered policy fields will auto-load and any drift between the registered plan "
            "and the as-run analysis will be flagged on the recommendation."
        )
        _manifest_upload = st.file_uploader(
            "Manifest JSON",
            type=["json"],
            key=f"manifest_upload_{key_suffix}",
            label_visibility="collapsed",
        )
        if _manifest_upload is not None:
            try:
                import json as _json
                _loaded = _json.loads(_manifest_upload.read())
                st.session_state["uploaded_manifest"] = _loaded
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
                    f"Policy fields applied; drift will be flagged on the result."
                )
            except Exception as _e:
                st.error(f"Could not parse manifest JSON: {_e}")
                st.session_state.pop("uploaded_manifest", None)
        elif (
            "uploaded_manifest" in st.session_state
            and st.button("Clear loaded manifest", key=f"clear_manifest_{key_suffix}")
        ):
            st.session_state.pop("uploaded_manifest", None)

        if "uploaded_manifest" in st.session_state:
            _m = st.session_state["uploaded_manifest"]
            st.info(
                f"📌 Active manifest: **{_m.get('experiment_id', 'unnamed')}** · "
                f"primary={_m.get('primary_metric','?')} · α={_m.get('alpha','?')} · "
                f"planned N={_m.get('planned_n_total','?')} · "
                f"toolkit v{_m.get('toolkit_version','?')}"
            )

    with st.expander("🎛️ Decision engine policy (optional)"):
        st.caption(
            "Tighten or relax the recommendation engine. Defaults match the documented "
            "decision matrices; change these only with stakeholder alignment (ideally "
            "captured in the manifest above)."
        )
        col_lt, col_swm = st.columns(2)
        with col_lt:
            _lt = st.number_input(
                "Expected-loss tolerance (Bayesian, absolute units)",
                min_value=0.0, value=float(st.session_state.get("loss_tolerance", 0.0) or 0.0),
                step=0.0005, format="%.5f",
                key=f"loss_tolerance_{key_suffix}",
                help="A 'Ship' is downgraded to Inconclusive when Bayesian expected loss exceeds this. "
                     "Leave 0 to disable.",
            )
            st.session_state["loss_tolerance"] = _lt
        with col_swm:
            _swm = st.checkbox(
                "Allow 'Ship with Monitoring' (intermediate state)",
                value=st.session_state.get("allow_ship_with_monitoring", False),
                key=f"allow_swm_{key_suffix}",
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
                key=f"mpt_{key_suffix}",
            )
            st.session_state["monitoring_prob_threshold"] = _mpt


# ── Data ingestion ──────────────────────────────────────────────────────────
if input_mode == "Upload CSV":
    # Clear manual state to prevent stale results from overriding CSV analysis
    st.session_state.pop("_manual_proportion", None)
    st.session_state.pop("_manual_continuous", None)
    with st.expander("📄 Expected CSV format"):
        st.markdown(
            "**Required**: `group` (`control` / `treatment`), `value` (0/1 for proportions, numeric otherwise).  \n"
            "**Optional**: `segment` (e.g. mobile/desktop), `covariate` (pre-experiment value for CUPED), "
            "`day` (integer day number from experiment start — enables novelty-effect detection)."
        )
    _render_preregistration_block(key_suffix="csv")
    st.markdown("---")
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

            st.caption(f"Loaded {len(df):,} rows · metric type: **{metric_type.value}** "
                       f"({'Z-test' if is_proportion else 'Welch t-test'})")
        except Exception as exc:
            display_error(str(exc))
else:
    section_header("Enter Summary Statistics", "Enter your experiment data", "✏️")

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

    st.markdown("---")
    _render_preregistration_block(key_suffix="manual")
    st.markdown("")

    if st.button(
        "🚀 Run Analysis",
        use_container_width=True,
        help=(
            "Summary-stats mode skips novelty and Simpson's Paradox checks — both need "
            "day- and segment-level data. Verify these upstream before aggregating "
            "(see the interview guide's 'Pre-Aggregation Checks' section)."
        ),
    ):
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
        step_names = ["Diagnostics", "Frequentist", "Bayesian", "Recommendation"]
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

        # SRM diagnostic
        srm_result = check_srm(observed=(_ct, _tt), expected_ratio=_manual_expected_ratio)
        render_diagnostics_card([_srm_check_to_dict(srm_result)])

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

        step_names = ["Diagnostics", "Frequentist", "Bayesian", "Recommendation"]
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

        # SRM diagnostic
        srm_result = check_srm(observed=(_cn, _tn), expected_ratio=_manual_expected_ratio)
        render_diagnostics_card([_srm_check_to_dict(srm_result)])

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

    # ── Compute all diagnostics up front, render unified card ──────────────
    with st.expander("⚙️ Allocation settings", expanded=False):
        srm_ratio_input = st.slider(
            "Planned allocation (% to control)",
            min_value=10, max_value=90, value=50, step=5,
            help="The intended traffic split. 50 = equal 50/50. 80 = 80% control / 20% treatment.",
        )
    srm_expected = (srm_ratio_input / 100, 1 - srm_ratio_input / 100)

    # Run all checks
    srm_result = check_srm(observed=(len(control), len(treatment)), expected_ratio=srm_expected)
    seg_result = segment_analysis(df) if has_segment else None
    novelty_result = check_novelty(df) if "day" in df.columns else None

    stratum_srm = None
    if "day" in df.columns:
        try:
            stratum_srm = check_srm_by_stratum(
                df, group_col="group", stratum_col="day",
                expected_ratio=srm_expected, threshold=0.01,
            )
        except Exception:
            stratum_srm = None

    _dup_count = int(df["unit_id"].duplicated().sum()) if "unit_id" in df.columns else 0
    _kurt = None
    if not is_proportion:
        from scipy.stats import kurtosis as _kurtosis
        _kurt = float(_kurtosis(df["value"].dropna(), fisher=True))

    # Build the diagnostics list
    diagnostics: list[dict] = []

    if "unit_id" in df.columns:
        diagnostics.append({
            "name": "Unit-of-analysis (one row per user)",
            "status": "pass" if _dup_count == 0 else "fail",
            "summary": (
                f"{_dup_count:,} duplicate unit_ids ({_dup_count / len(df):.1%})"
                if _dup_count > 0 else "no duplicates"
            ),
            "detail": (
                "Most A/B tests assume one row per user. Duplicates inflate effective N and "
                "shrink CIs. Aggregate to one row per unit (sum/mean as appropriate) before "
                "analysing."
            ) if _dup_count > 0 else None,
        })

    # SRM (aggregate)
    diagnostics.append(_srm_check_to_dict(srm_result))

    # Per-day SRM
    if stratum_srm is not None and stratum_srm.n_strata > 1:
        if stratum_srm.warning:
            diagnostics.append({
                "name": "Per-day traffic split",
                "status": "fail",
                "summary": (
                    f"{stratum_srm.n_mismatches}/{stratum_srm.n_strata} days flagged "
                    f"(Holm-corrected)"
                ),
                "detail": (
                    f"{stratum_srm.warning} Time-correlated SRM usually indicates a deploy, "
                    f"CDN switch, or filter rule that fired on certain days. The aggregate "
                    f"check can pass while this one fails."
                ),
            })
        else:
            diagnostics.append({
                "name": "Per-day traffic split",
                "status": "pass",
                "summary": f"{stratum_srm.n_strata} days, all clean (Holm)",
            })

    # Simpson's
    if seg_result is not None:
        if seg_result.simpsons_paradox:
            diagnostics.append({
                "name": "Simpson's Paradox",
                "status": "fail",
                "summary": "aggregate contradicts segments",
                "detail": (
                    f"{seg_result.simpsons_details} The headline result is misleading due to a "
                    f"composition effect across segments. Trust the within-segment effects."
                ),
            })
        else:
            diagnostics.append({
                "name": "Simpson's Paradox",
                "status": "pass",
                "summary": "segments consistent",
            })

    # Novelty
    if novelty_result is not None and novelty_result.details != "No day column — novelty check skipped." \
            and novelty_result.details != "Fewer than 3 days of data — novelty check skipped.":
        if novelty_result.has_novelty:
            diagnostics.append({
                "name": "Novelty effect",
                "status": "warn",
                "summary": f"early/late = {novelty_result.ratio:.2f}×",
                "detail": (
                    f"{novelty_result.details} Heuristic flag, not a statistical test — can "
                    f"produce false positives at low daily traffic. If real, re-analyse with "
                    f"the first 3–7 days excluded as a burn-in period."
                ),
            })
        else:
            diagnostics.append({
                "name": "Novelty effect",
                "status": "pass",
                "summary": f"early/late = {novelty_result.ratio:.2f}× (under 2.0× threshold)",
            })

    # Heavy-tail
    if _kurt is not None:
        if _kurt > 10:
            diagnostics.append({
                "name": "Heavy-tail metric",
                "status": "warn",
                "summary": f"excess kurtosis = {_kurt:.1f}",
                "detail": (
                    "A few extreme values dominate this metric — small number of outliers "
                    "carry most of the variance, making the test sensitive to chance and the "
                    "CI wide. Consider winsorising at the 99th percentile or analysing on a "
                    "log scale."
                ),
            })
        else:
            diagnostics.append({
                "name": "Heavy-tail metric",
                "status": "pass",
                "summary": f"excess kurtosis = {_kurt:.2f}",
            })

    render_diagnostics_card(diagnostics)

    # Optional per-day SRM details for power-users (collapsed, only when there are strata)
    if stratum_srm is not None and stratum_srm.n_strata > 1 and stratum_srm.stratum_results:
        with st.expander("📅 Per-day SRM details"):
            _strat_df = pd.DataFrame(stratum_srm.stratum_results)[
                ["stratum", "n_control", "n_treatment", "p_value", "p_value_adjusted", "has_mismatch"]
            ]
            st.dataframe(_strat_df, hide_index=True, use_container_width=True)

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

    # 5. Segmentation (optional) — seg_result already computed up front for the Diagnostics card
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
        # seg_result already computed up front for the Diagnostics card
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

    # novelty_result already computed up front for the Diagnostics card.
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
