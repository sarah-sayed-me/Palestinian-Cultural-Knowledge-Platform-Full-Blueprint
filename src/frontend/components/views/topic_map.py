"""Topic Map tab: Interactive scatter visualization of topics."""

import streamlit as st
from components.cards import render_section_header
from components.charts import create_topic_scatter


def render(data):
    """Render the topic map view.

    Args:
        data: output of services.backend.get_topic_data()
    """
    filters = data["filters"]
    topics = data["topics"]
    documents = data["documents"]
    topic_details = data["topic_details"]

    # Filter panel
    col1, col2, col3 = st.columns(3)
    with col1:
        source_filter = st.selectbox(
            "المصدر",
            filters["sources"],
            label_visibility="collapsed",
        )
    with col2:
        lang_filter = st.selectbox(
            "اللغة",
            filters["languages"],
            label_visibility="collapsed",
        )
    with col3:
        decade_filter = st.selectbox(
            "الفترة الزمنية",
            filters["decades"],
            label_visibility="collapsed",
        )

    # Apply filters
    filtered_docs = documents
    if source_filter != "الكل":
        filtered_docs = [d for d in filtered_docs if d["source"] == source_filter]
    if lang_filter != "الكل":
        filtered_docs = [d for d in filtered_docs if d["language"] == lang_filter]
    if decade_filter != "الكل":
        filtered_docs = [d for d in filtered_docs if d["decade"] == decade_filter]

    # Topic selection for highlighting
    topic_names = ["الكل"] + [t["name"] for t in topics]
    selected_topic_name = st.selectbox(
        "اختر موضوعًا للتفصيل",
        topic_names,
        label_visibility="collapsed",
    )

    highlight_id = None
    if selected_topic_name != "الكل":
        for t in topics:
            if t["name"] == selected_topic_name:
                highlight_id = t["id"]
                break

    # Render scatter chart
    fig = create_topic_scatter(filtered_docs, highlight_topic=highlight_id)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, theme=None)

    # Topic detail panel
    if highlight_id is not None and highlight_id in topic_details:
        detail = topic_details[highlight_id]
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        words_html = " ".join(
            f"<span style='background:rgba(0,151,54,0.1);color:#009736;padding:2px 8px;border-radius:12px;"
            f"font-size:0.82rem;margin:2px;display:inline-block;'>{w}</span>"
            for w in detail["top_words"]
        )
        docs_html = "<br>".join(f"\U0001f4d6 {doc}" for doc in detail["example_docs"])

        # words_html/docs_html are inlined onto the same line as an
        # adjacent tag (never alone on a line) — if either is "" (a topic
        # with no titled example docs, say), a lone blank line inside this
        # HTML block trips CommonMark's "blank line ends an HTML block"
        # rule and everything after renders as escaped literal text instead
        # of markup (same root cause fixed in kg_explorer.py/ask.py).
        st.markdown(
            f"""<div class='topic-detail-card'>
                <div style='font-size:1.1rem;font-weight:600;margin-bottom:0.75rem;color:#2C2C2C;'>
                    {detail['name']}
                </div>
                <div style='margin-bottom:0.5rem;'>
                    <span style='color:#8B8580;font-size:0.85rem;'>عدد الوثائق:</span>
                    <span style='font-weight:600;'>{detail['doc_count']}</span>
                </div>
                <div style='margin-bottom:0.75rem;'>
                    <span style='color:#8B8580;font-size:0.85rem;'>أكثر الكلمات ارتباطًا:</span><br>{words_html}
                </div>
                <div style='font-size:0.9rem;color:#555;line-height:1.8;'>{docs_html}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    elif not filtered_docs:
        st.markdown(
            """<div class='empty-state'>
                <div class='empty-state-icon'>\U0001f50d</div>
                <div class='empty-state-text'>لا توجد وثائق تطابق الفلاتر المحددة</div>
                <div class='empty-state-hint'>جرب تغيير معايير البحث</div>
            </div>""",
            unsafe_allow_html=True,
        )
