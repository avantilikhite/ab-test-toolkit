"""Streamlit-specific helper utilities."""

import streamlit as st


def get_alpha() -> float:
    """Get global alpha from session state."""
    return st.session_state.get("alpha", 0.05)


def get_practical_significance_threshold() -> float | None:
    """Get the practical-significance threshold (absolute units) from session state.

    When set, the recommendation engine emits a ``No Effect`` decision if the
    confidence interval is contained entirely within ±threshold. ``None`` means
    no practical-significance gate is applied.
    """
    val = st.session_state.get("practical_significance_threshold", 0.0)
    return float(val) if val and val > 0 else None


def get_loss_tolerance() -> float | None:
    """Read the expected-loss tolerance from session state. ``None`` disables the gate."""
    val = st.session_state.get("loss_tolerance", 0.0)
    return float(val) if val and val > 0 else None


def get_allow_ship_with_monitoring() -> bool:
    return bool(st.session_state.get("allow_ship_with_monitoring", False))


def render_sidebar_settings() -> None:
    """Render the global sidebar Settings block (Confidence Level + Practical-
    significance threshold) on every page.  Called from app.py and each page
    so users can adjust α and the practical-significance gate from anywhere
    in the multi-page app, not just the landing page.

    Also injects the prominent-selection CSS for sidebar selectbox/number_input
    and the larger tooltip-icon styling.
    """
    st.sidebar.markdown("### ⚙️ Settings")

    # CSS — flush-left to avoid markdown code-block escaping.
    st.sidebar.markdown(
        "<style>"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"select\"] > div {"
        "  background-color: #ffffff !important;"
        "  border: 2px solid #0066FF !important;"
        "  border-radius: 8px !important;"
        "  box-shadow: 0 1px 4px rgba(0, 102, 255, 0.18);"
        "}"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"select\"] *,"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"select\"] div[role=\"button\"],"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"select\"] div[role=\"button\"] *,"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"select\"] [data-baseweb=\"select-control\"] *,"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"select\"] span {"
        "  color: #0a1f4d !important;"
        "  font-weight: 700 !important;"
        "  font-size: 1.05rem !important;"
        "  letter-spacing: 0.3px;"
        "}"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"select\"] svg {"
        "  fill: #0066FF !important;"
        "  color: #0066FF !important;"
        "}"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"input\"] {"
        "  border: 2px solid #0066FF !important;"
        "  border-radius: 8px !important;"
        "  box-shadow: 0 1px 4px rgba(0, 102, 255, 0.18);"
        "}"
        "section[data-testid=\"stSidebar\"] div[data-baseweb=\"input\"] input {"
        "  color: #0a1f4d !important;"
        "  font-weight: 700 !important;"
        "  font-size: 1.05rem !important;"
        "  letter-spacing: 0.3px;"
        "}"
        "section[data-testid=\"stSidebar\"] [data-testid=\"stTooltipIcon\"],"
        "section[data-testid=\"stSidebar\"] [data-testid=\"stTooltipHoverTarget\"] {"
        "  color: #0066FF !important;"
        "  opacity: 1 !important;"
        "}"
        "section[data-testid=\"stSidebar\"] [data-testid=\"stTooltipIcon\"] svg,"
        "section[data-testid=\"stSidebar\"] [data-testid=\"stTooltipHoverTarget\"] svg {"
        "  width: 18px !important;"
        "  height: 18px !important;"
        "  fill: #0066FF !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )

    # Default to whatever was last picked, so the value persists across pages.
    _alpha_to_label = {0.10: "90%", 0.05: "95%", 0.01: "99%"}
    _current_alpha = float(st.session_state.get("alpha", 0.05))
    _default_label = _alpha_to_label.get(_current_alpha, "95%")
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        options=["90%", "95%", "99%"],
        index=["90%", "95%", "99%"].index(_default_label),
        help=(
            "Sets α — the chance you'll declare a winner when there's actually no real effect.\n\n"
            "**False positive** = your test says 'Treatment wins!' but the lift was just "
            "random noise. If you ship based on that, you may roll out a change that "
            "doesn't actually move the metric in production — and you've spent eng time, "
            "user trust, or money on nothing.\n\n"
            "• **90%** → α = 0.10 (1 in 10 'wins' are noise — sensitive but risky)\n"
            "• **95%** → α = 0.05 (1 in 20 'wins' are noise — industry default)\n"
            "• **99%** → α = 0.01 (1 in 100 'wins' are noise — very strict, requires bigger samples)\n\n"
            "Tighter confidence shrinks the false-positive risk but inflates the "
            "sample size you need to detect a real effect."
        ),
        key="_sidebar_confidence_level",
    )
    alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
    st.session_state["alpha"] = alpha_map[confidence_level]

    practical_threshold = st.sidebar.number_input(
        "Practical-significance threshold",
        min_value=0.0,
        value=float(st.session_state.get("practical_significance_threshold", 0.0) or 0.0),
        step=0.001,
        format="%.4f",
        help=(
            "Smallest effect size your business actually cares about, in absolute units.\n\n"
            "**Examples**\n"
            "• `0.005` = 0.5pp lift on a conversion-rate metric (e.g., 10% → 10.5%)\n"
            "• `0.50` = $0.50 change in **ARPU** (Average Revenue Per User — total "
            "revenue ÷ number of users in the experiment)\n\n"
            "**What it does:** when your confidence interval is fully inside ±this "
            "value, the engine reports **'No Effect'** instead of **'Inconclusive'** "
            "— meaning you've ruled out any business-meaningful impact, not just "
            "failed to detect one.\n\n"
            "**Set to 0 to turn this off:** the engine will then never return "
            "'No Effect' — non-significant results stay 'Inconclusive' regardless "
            "of how tight the CI is. Use 0 if you don't have a clear minimum effect "
            "threshold yet."
        ),
        key="_sidebar_practical_threshold",
    )
    st.session_state["practical_significance_threshold"] = practical_threshold


def render_header_credit() -> None:
    """Inject a fixed-position 'Avanti Likhite · LinkedIn · GitHub' credit
    into the top-right corner of the app, alongside Streamlit's header
    toolbar.  Also hides Streamlit Cloud's bottom-right 'Hosted with
    Streamlit' badge / Manage-app footer.  Call once per page (after
    st.set_page_config)."""
    # CSS first — must be flush-left so Streamlit's markdown parser doesn't
    # treat indented lines as a code block and escape the HTML.
    st.markdown(
        "<style>"
        "[data-testid=\"stStatusWidget\"] { display: none !important; }"
        ".viewerBadge_container__1QSob,"
        ".viewerBadge_link__1S137,"
        ".viewerBadge_text__1JaDK,"
        "a[href*=\"streamlit.io/cloud\"],"
        "a[href*=\"share.streamlit.io\"] { display: none !important; }"
        "footer { visibility: hidden !important; height: 0 !important; }"
        "#ab-toolkit-credit {"
        "  position: fixed !important;"
        "  top: 0.55rem !important;"
        "  right: 5rem !important;"
        "  z-index: 999999 !important;"
        "  font-size: 0.82rem;"
        "  color: #1A1A2E;"
        "  background: rgba(255, 255, 255, 0.95);"
        "  padding: 0.35rem 0.8rem;"
        "  border-radius: 6px;"
        "  border: 1px solid #cdd9f5;"
        "  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);"
        "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;"
        "}"
        "#ab-toolkit-credit a {"
        "  color: #0066FF !important;"
        "  text-decoration: none;"
        "  font-weight: 600;"
        "}"
        "#ab-toolkit-credit a:hover { text-decoration: underline; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div id="ab-toolkit-credit">'
        '<strong style="color:#0a1f4d;">Avanti Likhite</strong>'
        '&nbsp;·&nbsp;'
        '<a href="https://www.linkedin.com/in/avantilikhite" target="_blank">LinkedIn</a>'
        '&nbsp;·&nbsp;'
        '<a href="https://github.com/avantilikhite/ab-test-toolkit" target="_blank">GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )


def get_monitoring_prob_threshold() -> float:
    val = st.session_state.get("monitoring_prob_threshold", 0.85)
    return float(val) if val and val > 0 else 0.85


def get_uploaded_manifest() -> dict | None:
    """Return the parsed pre-reg manifest dict from session, or None."""
    return st.session_state.get("uploaded_manifest")


def display_error(message: str) -> None:
    """Display user-friendly error message."""
    st.error(f"⚠️ {message}")


def display_metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Display a metric in a card format."""
    st.metric(label=label, value=value, delta=delta)


def section_header(title: str, description: str = "", icon: str = "") -> None:
    """Render a styled section header with icon, title, description, and divider."""
    icon_str = f"{icon} " if icon else ""
    st.markdown(
        f"""<div style="margin-top:1.5rem;margin-bottom:0.5rem;">
            <h3 style="margin:0;color:#1A1A2E;">{icon_str}{title}</h3>
            {"<p style='margin:0.25rem 0 0 0;color:#555;font-size:0.95rem;'>" + description + "</p>" if description else ""}
        </div>""",
        unsafe_allow_html=True,
    )
    st.divider()


def metric_row(metrics: list[dict]) -> None:
    """Render a row of styled metric cards.

    Each dict should have keys: label, value, and optionally delta, icon, and help.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            icon = m.get("icon", "")
            label = m["label"]
            value = m["value"]
            delta = m.get("delta") or None
            help_text = m.get("help") or None
            display_label = f"{icon} {label}" if icon else label
            st.metric(label=display_label, value=value, delta=delta, help=help_text)


def status_badge(text: str, color: str = "blue") -> str:
    """Return HTML for an inline colored badge. Use with st.markdown(unsafe_allow_html=True)."""
    color_map = {
        "blue": ("#0066FF", "#e8f0ff"),
        "green": ("#0a8754", "#e6f7ef"),
        "red": ("#d32f2f", "#fde8e8"),
        "orange": ("#e67e22", "#fff4e5"),
        "gray": ("#555", "#f0f0f0"),
    }
    fg, bg = color_map.get(color, color_map["blue"])
    return (
        f'<span style="background:{bg};color:{fg};padding:0.2rem 0.65rem;'
        f'border-radius:12px;font-size:0.8rem;font-weight:600;">{text}</span>'
    )


def info_callout(text: str, callout_type: str = "info") -> None:
    """Render a styled callout box. Types: info, warning, success, error."""
    styles = {
        "info": ("#0066FF", "#eef4ff", "ℹ️"),
        "warning": ("#e67e22", "#fff8ed", "⚠️"),
        "success": ("#0a8754", "#edfaf3", "✅"),
        "error": ("#d32f2f", "#fef0f0", "❌"),
    }
    color, bg, emoji = styles.get(callout_type, styles["info"])
    st.markdown(
        f"""<div style="background:{bg};border-left:4px solid {color};border-radius:0 8px 8px 0;
            padding:0.9rem 1.2rem;margin:0.5rem 0;font-size:0.92rem;color:#333;">
            {emoji} {text}
        </div>""",
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a styled page header with icon, title, and subtitle."""
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#1A1A2E 0%,#16213e 100%);
            border-radius:12px;padding:1.8rem 2rem;margin-bottom:1.5rem;">
            <span style="font-size:2.2rem;">{icon}</span>
            <h1 style="margin:0.3rem 0 0 0;color:#fff;font-size:1.8rem;">{title}</h1>
            {"<p style='margin:0.4rem 0 0 0;color:#a0b0c8;font-size:1rem;'>" + subtitle + "</p>" if subtitle else ""}
        </div>""",
        unsafe_allow_html=True,
    )


def styled_container_open(border_color: str = "#e2e6ed") -> None:
    """Open a styled container div. Must pair with styled_container_close."""
    st.markdown(
        f"""<div style="border:1px solid {border_color};border-left:4px solid {border_color};
            border-radius:8px;padding:1.2rem 1.5rem;margin:0.8rem 0;background:#fafbfc;">""",
        unsafe_allow_html=True,
    )


def styled_container_close() -> None:
    """Close a styled container div."""
    st.markdown("</div>", unsafe_allow_html=True)
