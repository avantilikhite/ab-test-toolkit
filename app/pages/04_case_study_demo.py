"""📚 Case Study Demo — end-to-end walkthrough with synthetic data."""

import streamlit as st

from ab_test_toolkit.bayesian import beta_binomial
from ab_test_toolkit.cuped import cuped_adjust
from ab_test_toolkit.data_generator import generate_experiment_data
from ab_test_toolkit.frequentist import z_test
from ab_test_toolkit.recommendation import generate_recommendation, check_novelty
from ab_test_toolkit.segmentation import segment_analysis
from ab_test_toolkit.srm import check_srm
from ab_test_toolkit.visualization import (
    ci_comparison_plot,
    cumulative_lift_chart,
    daily_treatment_effect,
    mde_vs_n_curve,
    posterior_plot,
    power_loss_curve,
    segment_comparison_chart,
)
from ab_test_toolkit.power import power_curve

from app_utils import (
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

page_header("📚", "Case Study Demo", "End-to-end walkthrough with synthetic data and injected anomalies.")

st.markdown(
    """<div style="background:#eef4ff;border:1px solid #c8deff;border-radius:10px;
        padding:1.2rem 1.5rem;margin-bottom:1.5rem;">
        <p style="margin:0;color:#333;">
            This demo generates a <strong>synthetic A/B experiment</strong> with intentionally
            injected anomalies (novelty effect and Simpson's Paradox) and walks through
            every diagnostic check the toolkit provides.
        </p>
    </div>""",
    unsafe_allow_html=True,
)

alpha = get_alpha()

# ── Progress tracker ─────────────────────────────────────────────────────────
all_steps = [
    ("0", "Data Gen"),
    ("1", "SRM"),
    ("2", "Frequentist"),
    ("3", "Bayesian"),
    ("4", "CUPED"),
    ("5", "Segments"),
    ("6", "Verdict"),
    ("📈", "Charts"),
]
tracker_html = "".join(
    f'<span style="background:{"#0066FF" if s[0].isdigit() else "#00D4AA"};color:#fff;border-radius:50%;'
    f'width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;'
    f'font-size:0.7rem;font-weight:700;margin-right:0.25rem;">{s[0]}</span>'
    f'<span style="margin-right:1rem;font-size:0.82rem;color:#555;">{s[1]}</span>'
    for s in all_steps
)
st.markdown(
    f'<div style="background:#f8f9fb;border:1px solid #e2e6ed;border-radius:10px;'
    f'padding:0.8rem 1.2rem;margin-bottom:1.5rem;display:flex;align-items:center;flex-wrap:wrap;gap:0.2rem;">'
    f'{tracker_html}</div>',
    unsafe_allow_html=True,
)

# ── Step 0: Generate data ──────────────────────────────────────────────────
section_header("Data Generation", "Step 0 — Create synthetic experiment with injected anomalies", "🔬")
info_callout(
    "Creating 5,000 users per group with a 10% baseline conversion rate and a "
    "2pp absolute lift. Novelty effect and Simpson's Paradox are injected.",
    "info",
)

df = generate_experiment_data(
    baseline_rate=0.10,
    effect_size=0.02,
    n_control=5000,
    n_treatment=5000,
    inject_novelty=True,
    inject_simpsons=True,
)
with st.expander("📄 Preview dataset", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
st.markdown(
    f'<p style="color:#888;font-size:0.82rem;">Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns</p>',
    unsafe_allow_html=True,
)

control = df.loc[df["group"] == "control", "value"].to_numpy()
treatment = df.loc[df["group"] == "treatment", "value"].to_numpy()

# ── Step 1: SRM check ──────────────────────────────────────────────────────
section_header("Sample Ratio Mismatch", "Step 1 — Verify traffic was split correctly", "🔄")
info_callout(
    "Before looking at any results, verify that traffic was split correctly. "
    "An SRM would indicate a bug in the randomisation layer.",
    "info",
)
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
        f"SRM detected — χ² = {srm_result.chi2_statistic:.2f}, p = {srm_result.p_value:.4f}",
        "warning",
    )
else:
    info_callout(
        f"No SRM — χ² = {srm_result.chi2_statistic:.2f}, p = {srm_result.p_value:.4f}. Traffic split is clean.",
        "success",
    )

# ── Step 2: Frequentist analysis ───────────────────────────────────────────
section_header("Frequentist Analysis (Z-test)", "Step 2 — p-value and confidence interval", "📐")
info_callout(
    "A two-proportion Z-test compares the conversion rates and produces a "
    "p-value and confidence interval for the difference.",
    "info",
)
freq_result = z_test(control, treatment, alpha=alpha)

sig_badge = status_badge("✅ Significant", "green") if freq_result.is_significant else status_badge("❌ Not Significant", "red")
st.markdown(
    f"**p-value**: {freq_result.p_value:.4f} &nbsp;|&nbsp; {sig_badge} at α = {alpha}",
    unsafe_allow_html=True,
)
st.markdown(f"**Effect size** (Cohen's h): {freq_result.effect_size:.4f}")
st.plotly_chart(ci_comparison_plot(freq_result), use_container_width=True)

# ── Step 3: Bayesian analysis ──────────────────────────────────────────────
section_header("Bayesian Analysis (Beta-Binomial)", "Step 3 — Posterior probability for stakeholders", "🎲")
bayes_result = beta_binomial(control, treatment)

metric_row([
    {"icon": "🎲", "label": "P(Treatment > Control)", "value": f"{bayes_result.prob_b_greater_a:.2%}",
     "help": "Posterior probability — answers: 'Given the data, what is the probability Treatment is truly better than Control?' Unlike a p-value (which assumes no difference and asks how surprising the data is), this directly tells you the likelihood B beats A. Above 95% is typically strong evidence."},
    {"icon": "📉", "label": "Expected Loss", "value": f"{bayes_result.expected_loss:.5f}",
     "help": "The expected loss (regret) if you choose treatment and it turns out to be worse than control. Lower is better — a value close to 0 means little downside risk to shipping the treatment."},
])
st.markdown("")
st.plotly_chart(posterior_plot(bayes_result), use_container_width=True)

# ── Step 4: CUPED ──────────────────────────────────────────────────────────
section_header("CUPED Variance Reduction", "Step 4 — Pre-experiment covariate adjustment", "✨")
info_callout(
    "When a pre-experiment covariate is available, CUPED reduces the variance of "
    "the treatment-effect estimate, tightening the confidence interval.",
    "info",
)
if "covariate" in df.columns:
    ctrl_cov = df.loc[df["group"] == "control", "covariate"].to_numpy()
    treat_cov = df.loc[df["group"] == "treatment", "covariate"].to_numpy()
    cuped_result = cuped_adjust(control, treatment, ctrl_cov, treat_cov, alpha=alpha)

    metric_row([
        {"icon": "📊", "label": "Variance Reduction", "value": f"{cuped_result.variance_reduction_pct:.1f}%"},
        {"icon": "📏", "label": "Unadjusted Estimate", "value": f"{cuped_result.unadjusted_estimate:.4f}"},
        {"icon": "✨", "label": "CUPED-Adjusted Estimate", "value": f"{cuped_result.adjusted_estimate:.4f}"},
    ])
    st.markdown("")
    info_callout(
        f"Unadjusted CI: [{cuped_result.unadjusted_ci[0]:.4f}, {cuped_result.unadjusted_ci[1]:.4f}] → "
        f"Adjusted CI: [{cuped_result.adjusted_ci[0]:.4f}, {cuped_result.adjusted_ci[1]:.4f}]",
        "success",
    )
else:
    info_callout("No covariate column found — skipping CUPED.", "info")

# ── Step 5: Segmentation ──────────────────────────────────────────────────
section_header("Segment Analysis", "Step 5 — Heterogeneous treatment effects and Simpson's Paradox", "🔎")
info_callout(
    "Breaking results down by segment can reveal heterogeneous treatment effects "
    "and Simpson's Paradox — where the overall result disagrees with every segment-level result.",
    "info",
)
if "segment" in df.columns:
    seg_result = segment_analysis(df)
    if seg_result.simpsons_paradox:
        info_callout(f"Simpson's Paradox: {seg_result.simpsons_details}", "warning")
    for seg in seg_result.segment_results:
        adj_p = seg.get("p_value_adjusted", seg["p_value"])
        sig_color = "green" if adj_p < alpha else "gray"
        badge = status_badge(f"p={seg['p_value']:.4f} (adj={adj_p:.4f})", sig_color)
        st.markdown(
            f"**{seg['segment']}** — effect: {seg['estimate']:.4f}, "
            f"CI: [{seg['ci'][0]:.4f}, {seg['ci'][1]:.4f}] {badge}",
            unsafe_allow_html=True,
        )
    st.plotly_chart(segment_comparison_chart(seg_result), use_container_width=True)
else:
    seg_result = None
    info_callout("No segment column found — skipping segmentation.", "info")

# ── Step 6: Recommendation ─────────────────────────────────────────────────
section_header("Recommendation", "Step 6 — Final verdict from all evidence", "🏁")
info_callout(
    "The recommendation engine synthesises frequentist significance, Bayesian "
    "probability, SRM status, segment-level flags, and novelty detection into a single verdict.",
    "info",
)

novelty_result = None
if "day" in df.columns:
    novelty_result = check_novelty(df)
    if novelty_result.has_novelty:
        info_callout(novelty_result.details, "warning")
    else:
        info_callout(
            f"No novelty effect detected (early/late ratio: {novelty_result.ratio:.2f}).",
            "success",
        )

rec = generate_recommendation(
    frequentist=freq_result,
    bayesian=bayes_result,
    srm=srm_result,
    segmentation=seg_result if "segment" in df.columns else None,
    novelty=novelty_result,
    has_covariate="covariate" in df.columns,
    practical_significance_threshold=get_practical_significance_threshold(),
    loss_tolerance=get_loss_tolerance(),
    allow_ship_with_monitoring=get_allow_ship_with_monitoring(),
    monitoring_prob_threshold=get_monitoring_prob_threshold(),
    manifest=get_uploaded_manifest(),
)
color_map = {
    "Ship": "green", "Ship with Monitoring": "blue",
    "No-Ship": "red", "Inconclusive": "orange", "No Effect": "gray",
}
bg_map = {
    "Ship": "#edfaf3", "Ship with Monitoring": "#eef4ff",
    "No-Ship": "#fef0f0", "Inconclusive": "#fff8ed", "No Effect": "#f5f5f5",
}
border_map = {
    "Ship": "#0a8754", "Ship with Monitoring": "#0066FF",
    "No-Ship": "#d32f2f", "Inconclusive": "#e67e22", "No Effect": "#888",
}
emoji_map = {
    "Ship": "🚀", "Ship with Monitoring": "🛟",
    "No-Ship": "🛑", "Inconclusive": "🤔", "No Effect": "⚖️",
}
color = color_map.get(rec.recommendation, "gray")
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

with st.expander("📋 Supporting metrics"):
    st.json(rec.supporting_metrics)

# ── Bonus Charts ────────────────────────────────────────────────────────────
st.markdown("")
section_header("Additional Charts", "Bonus visualizations for deeper insight", "📈")

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown("##### MDE vs. Sample Size")
    pc_data = power_curve(baseline_rate=0.10, mde_range=(0.005, 0.10), alpha=alpha)
    st.plotly_chart(mde_vs_n_curve(pc_data), use_container_width=True)

with col_b:
    st.markdown("##### Sample-Size Inflation vs. Allocation Ratio")
    st.plotly_chart(
        power_loss_curve(baseline_rate=0.10, mde=0.02, alpha=alpha),
        use_container_width=True,
    )

if "day" in df.columns:
    st.divider()
    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        st.markdown("##### Cumulative Lift")
        st.plotly_chart(cumulative_lift_chart(df), use_container_width=True)
    with col_d:
        st.markdown("##### Daily Treatment Effect")
        st.plotly_chart(daily_treatment_effect(df), use_container_width=True)
