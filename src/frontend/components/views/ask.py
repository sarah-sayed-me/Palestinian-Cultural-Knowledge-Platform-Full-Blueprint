"""RAG Ask tab: Question-answering interface with citations."""

import streamlit as st


SUGGESTED_QUESTIONS = [
    "ما هي أهم عناصر التراث الثقافي الفلسطيني؟",
    "ما هو التطريز الفلسطيني وما دلالته الثقافية؟",
    "ما أهم المدن الفلسطينية المذكورة في المصادر؟",
    "ما أبرز الموضوعات الثقافية في corpus؟",
]


def render(ask_fn):
    """Render the Ask / RAG view.

    Args:
        ask_fn: callable, services.backend.ask_question
    """
    # Header
    st.markdown(
        """<div style='text-align:center;margin-bottom:1.5rem;'>
            <div style='font-size:1.15rem;font-weight:600;color:#2C2C2C;margin-bottom:0.3rem;'>
                اسأل المنصة عن الثقافة والتاريخ والتراث الفلسطيني
            </div>
            <div style='font-size:0.85rem;color:#8B8580;'>
                يستخدم النظام RAG لاسترجاع الإجابات من قاعدة المعرفة مع الاستشهاد بالمصادر
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # A suggestion click from the *previous* run is applied here —
    # session_state for a widget's key can only be set before that widget is
    # instantiated in a given run, so this has to land ahead of
    # st.text_input(key="ask_input") below. The suggestion button itself
    # (further down) only stashes the pending value and calls st.rerun();
    # it cannot set "ask_input" directly since that widget is already
    # instantiated by the time its click is handled in the same run.
    pending_question = st.session_state.pop("_pending_question", None)
    auto_ask = pending_question is not None
    if pending_question is not None:
        st.session_state["ask_input"] = pending_question

    # Question input
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "",
            placeholder="اكتب سؤالك هنا...",
            label_visibility="collapsed",
            key="ask_input",
        )
    with col2:
        st.markdown("<div style='height:2.4rem'></div>", unsafe_allow_html=True)
        ask_clicked = st.button("اسأل", use_container_width=True, type="primary")

    # Suggested questions
    st.markdown("<div style='color:#8B8580;font-size:0.82rem;margin-bottom:0.4rem;'>أسئلة مقترحة:</div>", unsafe_allow_html=True)
    sug_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, sq in enumerate(SUGGESTED_QUESTIONS):
        with sug_cols[i]:
            if st.button(sq, key=f"sug_{i}", use_container_width=True):
                st.session_state["_pending_question"] = sq
                st.rerun()

    # Process question
    if question and (ask_clicked or auto_ask):
        # Loading state
        with st.spinner(""):
            st.markdown(
                """<div class='loading-container'>
                    <div style='margin-bottom:0.75rem;color:#556B2F;font-size:0.95rem;'>
                        جاري البحث في قاعدة المعرفة...
                    </div>
                    <div class='loading-dots'>
                        <span></span><span></span><span></span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Call backend
            result = ask_fn(question)

        # Show answer
        answer_text = result["answer"]
        citations = result.get("citations", [])
        metadata = result.get("metadata", {})

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Answer card
        citations_html = ""
        if citations:
            cite_badges = []
            for c in citations:
                if c.get("url"):
                    cite_badges.append(
                        f"<a href='{c['url']}' target='_blank' class='citation-badge'>"
                        f"[{c['id']}] {c['source']}</a>"
                    )
                else:
                    cite_badges.append(
                        f"<span class='citation-badge'>[{c['id']}] {c['source']}</span>"
                    )
            citations_html = (
                "<div style='margin-top:1rem;'>"
                "<span style='color:#8B8580;font-size:0.85rem;'>المصادر:</span> "
                + " ".join(cite_badges)
                + "</div>"
            )

        meta_html = ""
        if metadata:
            meta_items = []
            if "sources_used" in metadata:
                meta_items.append(f"المصادر المستخدمة: <b>{metadata['sources_used']}</b>")
            if "chunks_retrieved" in metadata:
                meta_items.append(f"عدد النتائج: <b>{metadata['chunks_retrieved']}</b>")
            if "response_time" in metadata:
                meta_items.append(f"زمن الاستجابة: <b>{metadata['response_time']}</b>")
            meta_html = (
                "<div class='metadata-row'>"
                + "".join(f"<span class='metadata-item'>{m}</span>" for m in meta_items)
                + "</div>"
            )

        # citations_html/meta_html are inlined onto the same line as
        # neighboring tags rather than each on its own line — when either is
        # "" (no citations, or the error path's empty metadata), a lone
        # blank line inside this HTML block trips CommonMark's "blank line
        # ends an HTML block" rule and everything after renders as escaped
        # literal text instead of markup (same root cause fixed in
        # kg_explorer.py's entity panel).
        st.markdown(
            f"""<div class='answer-card'>
                <div style='font-size:0.9rem;line-height:2;color:#2C2C2C;'>{answer_text}</div>
                {citations_html}</div>{meta_html}""",
            unsafe_allow_html=True,
        )
