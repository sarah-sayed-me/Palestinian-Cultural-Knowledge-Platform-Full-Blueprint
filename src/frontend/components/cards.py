"""Reusable card and metric components."""

import streamlit as st


PALESTINIAN_COLORS = [
    "#009736",  # Palestinian green
    "#CE1126",  # Palestinian red
    "#C5A55A",  # Warm gold
    "#6B8E23",  # Olive green
    "#556B2F",  # Dark olive
    "#1a1a1a",  # Black
    "#8B8580",  # Warm gray
]


def render_kpi_cards(kpis: dict):
    """Render the 4 main KPI metric cards in a row.

    Args:
        kpis: dict with keys total_documents, sources, kg_entities, kg_relations
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-icon">📄</div>
                <div class="kpi-value">{kpis['total_documents']:,}</div>
                <div class="kpi-label">إجمالي الوثائق</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-icon">🗂️</div>
                <div class="kpi-value">{kpis['sources']}</div>
                <div class="kpi-label">المصادر</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-icon">⚙️</div>
                <div class="kpi-value">{kpis['kg_entities']:,}</div>
                <div class="kpi-label">كيانات الرسم المعرفي</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-icon">🔗</div>
                <div class="kpi-value">{kpis['kg_relations']:,}</div>
                <div class="kpi-label">العلاقات</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_section_header(title: str, icon: str = ""):
    """Render a styled section header with optional icon."""
    if icon:
        st.markdown(
            f"""<div class="section-header">
                <span class="header-icon">{icon}</span>
                {title}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="section-header">{title}</div>""",
            unsafe_allow_html=True,
        )


def render_content_card(title: str, content_html: str):
    """Render a generic content card with title and HTML body."""
    st.markdown(
        f"""<div class="content-card">
            <div class="content-card-header">{title}</div>
            {content_html}
        </div>""",
        unsafe_allow_html=True,
    )


def render_empty_state(icon: str, text: str, hint: str = ""):
    """Render an empty state placeholder."""
    st.markdown(
        f"""<div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-text">{text}</div>
            <div class="empty-state-hint">{hint}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_source_badge(source_name: str):
    """Render a small source badge."""
    st.markdown(
        f"<span class='source-badge'>🌐 {source_name}</span>",
        unsafe_allow_html=True,
    )
