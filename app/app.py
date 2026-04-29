"""AB Test Toolkit — Streamlit App Entry Point."""

import streamlit as st

from app_utils import render_header_credit, render_sidebar_settings

st.set_page_config(
    page_title="AB Test Toolkit",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header_credit()

# ── Custom CSS theme ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* --- Typography & base --- */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }

    /* --- Sidebar branding --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown h1 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #a0b0c8 !important;
    }
    /* --- Sidebar navigation links --- */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"],
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] span,
    [data-testid="stSidebar"] nav a,
    [data-testid="stSidebar"] nav span {
        color: #e0e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover,
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.08) !important;
    }

    /* --- Metric cards --- */
    [data-testid="stMetric"] {
        background: #f8f9fb;
        border: 1px solid #e2e6ed;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        overflow: visible !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #777 !important;
        letter-spacing: 0.3px;
    }
    [data-testid="stMetricValue"] {
        color: #1A1A2E !important;
        font-weight: 700 !important;
    }

    /* --- Dividers --- */
    [data-testid="stHorizontalRule"] {
        border-color: #e8ecf1 !important;
    }

    /* --- Plotly charts --- */
    .stPlotlyChart {
        border: 1px solid #eee;
        border-radius: 10px;
        overflow: hidden;
    }

    /* --- Expanders --- */
    [data-testid="stExpander"] {
        border: 1px solid #e2e6ed;
        border-radius: 10px;
        overflow: hidden;
    }

    /* --- Page link cards --- */
    a[data-testid="stPageLink-NavLink"] {
        background: #f8f9fb !important;
        border: 1px solid #e2e6ed !important;
        border-radius: 12px !important;
        padding: 1.2rem 1rem !important;
        min-height: 120px !important;
        transition: all 0.2s ease !important;
        text-decoration: none !important;
    }
    a[data-testid="stPageLink-NavLink"]:hover {
        background: #eef4ff !important;
        border-color: #0066FF !important;
        box-shadow: 0 4px 12px rgba(0,102,255,0.1) !important;
        transform: translateY(-2px);
    }
    a[data-testid="stPageLink-NavLink"] p {
        color: #333 !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center;padding:0.5rem 0 1rem 0;">
        <span style="font-size:2rem;">🧪</span>
        <h2 style="margin:0.2rem 0 0 0;font-size:1.2rem;color:#fff !important;">AB Test Toolkit</h2>
        <p style="margin:0;font-size:0.75rem;color:#6c8bad !important;">Production-Grade Experimentation</p>
    </div>
    <hr style="border-color:#2a3a5e;margin:0 0 1rem 0;">
    """,
    unsafe_allow_html=True,
)
render_sidebar_settings()

# ── Landing page ─────────────────────────────────────────────────────────────

# Hero section
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1A1A2E 0%,#0f3460 50%,#0066FF 100%);
        border-radius:16px;padding:3rem 2.5rem;margin-bottom:2rem;text-align:center;">
        <span style="font-size:3rem;">🧪</span>
        <h1 style="color:#fff;margin:0.5rem 0 0.3rem 0;font-size:2.2rem;">AB Test Toolkit</h1>
        <p style="color:#a0c4ff;font-size:1.1rem;margin:0;max-width:600px;display:inline-block;">
            A production-grade tool for designing, auditing, and analyzing A/B experiments
            with statistical rigor.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature cards — fully clickable navigation
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link(
        "pages/01_experiment_design.py",
        label="🎯 **Experiment Design**\n\nCalculate sample sizes & power trade-offs",
        use_container_width=True,
    )
with col2:
    st.page_link(
        "pages/02_analyze_results.py",
        label="📊 **Analyze Results**\n\nUpload CSV or enter stats for full analysis",
        use_container_width=True,
    )
with col3:
    st.page_link(
        "pages/03_sensitivity_analysis.py",
        label="🔍 **Sensitivity Analysis**\n\nPost-experiment MDE detection",
        use_container_width=True,
    )
with col4:
    st.page_link(
        "pages/04_case_study_demo.py",
        label="📚 **Case Study Demo**\n\nEnd-to-end walkthrough with diagnostics",
        use_container_width=True,
    )

# Getting started
st.markdown(
    """
    <div style="background:#eef4ff;border:1px solid #c8deff;border-radius:12px;padding:1.5rem 2rem;margin-bottom:2rem;">
        <h4 style="margin:0 0 0.8rem 0;color:#0066FF;">🚀 Getting Started</h4>
        <ol style="margin:0;padding-left:1.2rem;color:#333;line-height:1.8;">
            <li><strong>Set your confidence level</strong> in the sidebar (default: 95%)</li>
            <li><strong>Navigate to a page</strong> using the sidebar menu</li>
            <li>Start with <strong>Experiment Design</strong> to plan, or jump to <strong>Analyze Results</strong> with your data</li>
            <li>Try the <strong>Case Study Demo</strong> for a complete walkthrough</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    """
    <div style="text-align:center;padding:1.5rem 0 0.5rem 0;color:#aaa;font-size:0.78rem;">
        <hr style="border-color:#eee;margin-bottom:1rem;">
        🧪 AB Test Toolkit · Built with Streamlit · Statistical rigor made simple
    </div>
    """,
    unsafe_allow_html=True,
)
