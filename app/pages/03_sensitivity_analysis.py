"""🔍 Sensitivity Analysis — post-experiment minimum detectable effect."""

import streamlit as st

from ab_test_toolkit.power import power_curve

from app_utils import (
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

page_header(
    "🔍",
    "Sensitivity Analysis (Conversion / Proportion Metrics)",
    "Post-experiment minimum detectable effect at 80% power. Continuous metrics are not yet supported on this page — they are on the roadmap."
)

with st.container():
    st.markdown(
        """<div style="background:#eef4ff;border:1px solid #c8deff;border-radius:10px;
            padding:1.2rem 1.5rem;margin-bottom:1.5rem;">
            <p style="margin:0 0 0.5rem 0;color:#333;">
                After an experiment concludes, a common question is:
                <em>"Given my sample size, what was the smallest effect I could have detected?"</em>
            </p>
            <p style="margin:0;color:#333;">
                This page computes the <strong>Minimum Detectable Effect (MDE)</strong>
                at 80% power for the observed sample size.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📖 Why not 'observed power'?"):
        st.markdown(
            """
            **Note:** This is *not* the same as "observed power" (which is a function
            of the observed effect and is therefore circular). Instead, we fix power
            at 80% and solve for the MDE — the recommended approach per
            Hoenig & Heisey (2001).
            """
        )

    with st.expander("💡 What does 80% power mean?"):
        st.markdown(
            """
            **Power = 80%** means that if a true effect of exactly the MDE size exists,
            the experiment has an **80% chance of detecting it** (i.e., returning a
            statistically significant result) and a **20% chance of missing it**
            (a Type II error / false negative).

            In practical terms: if the MDE shown below is 1.8pp and the real effect
            is 1.8pp, you would expect **4 out of 5** identically designed experiments
            to correctly flag it as significant. The fifth would return "not significant"
            purely due to random noise — not because the effect isn't real.

            **Why 80%?** It's the industry-standard balance between detection
            sensitivity and sample cost. Higher power (e.g., 90%) requires
            ~30% more sample, which means longer experiments and higher
            opportunity cost.
            """
        )

# --- Inputs ---
section_header("Parameters", "Enter your observed experiment data", "⚙️")
col1, col2 = st.columns(2, gap="large")
with col1:
    observed_n = st.number_input(
        "Observed sample size per group",
        min_value=100,
        max_value=10_000_000,
        value=5000,
        step=500,
        help="Total number of observations in each group (control or treatment).",
    )
with col2:
    baseline_rate = st.slider(
        "Baseline conversion rate",
        min_value=0.0001,
        max_value=0.99,
        value=0.10,
        step=0.0005,
        format="%.4f",
        help="Your control group's conversion rate. Supports rare-event baselines down to 0.01%.",
    )

alpha = get_alpha()

# --- Compute MDE via power curve inversion ---
try:
    # Cap upper MDE to keep p1 = baseline + mde < 1
    max_valid_mde = (1.0 - baseline_rate) * 0.95
    # For low baselines, widen the search range beyond baseline_rate
    mde_upper = min(max(baseline_rate * 2, 0.05), max_valid_mde)
    pc = power_curve(
        baseline_rate=baseline_rate,
        mde_range=(0.001, mde_upper),
        alpha=alpha,
        n_points=200,
    )

    feasible = pc[pc["n_control"] <= observed_n]
    if feasible.empty:
        info_callout(
            "The observed sample size is smaller than the minimum required for any "
            "MDE in the search range. Consider collecting more data.",
            "warning",
        )
    else:
        best_row = feasible.iloc[0]  # smallest MDE that is still detectable
        detected_mde = best_row["mde"]
        relative_mde = detected_mde / baseline_rate * 100

        st.markdown("")
        section_header("Result", "Minimum detectable effect for your sample", "🎯")

        metric_row([
            {"icon": "👥", "label": "Observed N / group", "value": f"{observed_n:,}"},
            {"icon": "📏", "label": "MDE (absolute)", "value": f"{detected_mde:.4f}"},
            {"icon": "📐", "label": "MDE (relative)", "value": f"{relative_mde:.1f}%"},
        ])

        st.markdown("")
        info_callout(
            f"With <strong>{observed_n:,}</strong> observations per group and a baseline rate of "
            f"<strong>{baseline_rate:.2%}</strong>, the experiment could reliably detect an absolute "
            f"effect of <strong>{detected_mde:.4f}</strong> (≈ {relative_mde:.1f}% relative) at "
            f"80% power and α = {alpha}.",
            "success",
        )

except Exception as exc:
    st.error(f"⚠️ Computation error: {exc}")
