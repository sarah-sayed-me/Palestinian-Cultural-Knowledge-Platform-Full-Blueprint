"""Bias Meter tab: Content representation analysis across sources."""

import streamlit as st
from components.charts import (
    create_bias_gauge,
    create_bias_source_comparison,
    create_bias_dimension_radar,
)


def render(data):
    """Render the bias meter view.

    Args:
        data: output of services.backend.get_bias_data()
    """
    filters = data["filters"]
    dimensions = data["dimension_scores"]
    source_comparison = data["source_comparison"]
    stacked_categories = data.get("stacked_categories")
    weat_score = data["weat_score"]
    weat_desc = data["weat_description"]

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("المصدر", filters["sources"], label_visibility="collapsed")
    with col2:
        st.selectbox("اللغة", filters["languages"], label_visibility="collapsed")
    with col3:
        st.selectbox("الفترة الزمنية", filters["decades"], label_visibility="collapsed")

    # WEAT explanation
    st.markdown(
        f"""<div class='content-card'>
            <div class='content-card-header'>مقياس WEAT لتقييم التحيز التضميني</div>
            <div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;'>
                <span style='font-size:2rem;font-weight:700;color:{"#CE1126" if weat_score < 0 else "#009736"};'>{weat_score:.3f}</span>
                <span style='color:#8B8580;font-size:0.85rem;'>effect size</span>
            </div>
            <div style='font-size:0.9rem;color:#555;line-height:1.8;'>{weat_desc}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not dimensions:
        st.info("لا توجد بيانات تصنيف محتوى بعد — شغّل scripts/run_content_classification.py ثم scripts/run_bias_measurement.py.")
        return

    # Gauges grid — st.plotly_chart(use_container_width=True) scales the
    # whole figure (numbers included) to fit its column, so packing all 6
    # into one row left each too narrow to read clearly (only looked "clear"
    # in Streamlit's fullscreen view, which gives a chart the full page
    # width instead of 1/6 of it). 3 per row roughly doubles each gauge's
    # width and, with it, how large the percentage renders.
    GAUGES_PER_ROW = 3
    for row_start in range(0, len(dimensions), GAUGES_PER_ROW):
        row_dims = dimensions[row_start:row_start + GAUGES_PER_ROW]
        gauge_cols = st.columns(GAUGES_PER_ROW)
        for col, dim in zip(gauge_cols, row_dims):
            with col:
                fig = create_bias_gauge(
                    dim["dimension"],
                    dim["value"],
                    "ثقافة",
                    "صراع",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Source comparison stacked bar + radar
    col1, col2 = st.columns(2)
    with col1:
        fig = create_bias_source_comparison(source_comparison, categories=stacked_categories)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
    with col2:
        fig = create_bias_dimension_radar(dimensions)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)

    # Dimension details with tooltips
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    for dim in dimensions:
        sources_text = " | ".join(
            f"{src}: {val}%" for src, val in dim["sources"].items()
        )
        with st.expander(f"{dim['dimension']} ({dim['dimension_en']}) — {dim['value']}%"):
            st.markdown(f"**الوصف:** {dim['description']}")
            st.markdown(f"**التوزيع حسب المصدر:** {sources_text}")
