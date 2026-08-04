"""Timeline tab: Historical document distribution and term frequency analysis."""

import streamlit as st
from components.charts import create_timeline_bar, create_term_frequency_line


def render(data):
    """Render the timeline view.

    Args:
        data: output of services.backend.get_timeline_data()
    """
    docs_by_decade = data["docs_by_decade"]
    available_terms = data["available_terms"]
    term_frequencies = data["term_frequencies"]
    era_info = data["era_info"]

    # Timeline bar chart — deliberately corpus-wide (every document, not
    # just the selected term below), so it stays constant as the term
    # selector changes; that's by design, not a stale/non-reactive chart.
    fig = create_timeline_bar(docs_by_decade)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
    st.caption("يعرض هذا الرسم توزيع **كل** وثائق المجموعة عبر الزمن (ثابت)، بصرف النظر عن المصطلح المختار أدناه — أما تكرار المصطلح المحدد فيظهر في الرسم البياني الثاني.")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Term frequency section
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_term = st.selectbox(
            "اختر مصطلحًا ثقافيًا",
            available_terms,
            label_visibility="collapsed",
        )
    with col2:
        show_comparison = st.checkbox("مقارنة مع مصطلح آخر", value=False)

    freq_data = term_frequencies.get(selected_term, [])
    extra = []
    if show_comparison:
        other_terms = [t for t in available_terms if t != selected_term]
        if other_terms:
            compare_term = st.selectbox(
                "مصطلح المقارنة",
                other_terms,
                label_visibility="collapsed",
            )
            extra.append((compare_term, term_frequencies.get(compare_term, [])))

    fig = create_term_frequency_line(selected_term, freq_data, extra_terms=extra if extra else None)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)

    # Era info panel
    if freq_data and era_info:
        mid_year = freq_data[len(freq_data) // 2]["year"]
        closest_era = min(era_info.keys(), key=lambda k: abs(int(k) - mid_year))
        era_text = era_info[closest_era]
        st.markdown(
            f"""<div class='timeline-era-card'>
                <div style='font-weight:600;color:#009736;margin-bottom:0.3rem;'>الفترة: {closest_era}s</div>
                <div style='font-size:0.9rem;color:#555;line-height:1.8;'>{era_text}</div>
            </div>""",
            unsafe_allow_html=True,
        )
