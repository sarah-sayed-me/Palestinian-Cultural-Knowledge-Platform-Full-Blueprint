"""Overview tab: Dashboard with KPIs and distribution charts."""

import streamlit as st
from components.cards import render_kpi_cards
from components.charts import (
    create_source_bar_chart,
    create_language_donut,
    create_topic_bar,
    create_entity_type_chart,
)


def render(data):
    """Render the overview dashboard.

    Args:
        data: output of services.backend.get_overview_data()
    """
    render_kpi_cards(data["kpis"])

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = create_source_bar_chart(data["documents_by_source"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
    with col2:
        fig = create_language_donut(data["documents_by_language"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)

    col1, col2 = st.columns(2)
    with col1:
        fig = create_topic_bar(data["topic_distribution"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
    with col2:
        fig = create_entity_type_chart(data["kg_entity_types"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
