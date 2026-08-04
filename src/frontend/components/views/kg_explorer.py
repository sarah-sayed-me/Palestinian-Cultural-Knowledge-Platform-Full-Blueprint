"""Knowledge Graph Explorer tab: Search entities and visualize graph neighborhoods.

This component is isolated so the visualization implementation (Plotly/PyVis/NetworkX)
can be swapped independently.
"""

import streamlit as st
from components.charts import create_kg_plotly_graph


def render(data, graph_search_fn):
    """Render the KG explorer view.

    Args:
        data: output of services.backend.get_kg_data()
        graph_search_fn: callable, services.backend.search_knowledge_graph
    """
    stats = data["stats"]
    sample_entities = data["sample_entities"]

    # A chip click from the *previous* run is applied here — same
    # stash-then-rerun pattern as the Ask tab's suggested questions, because
    # session_state for a widget's key can only be set before that widget is
    # instantiated in a given run, and "kg_query" is about to be
    # instantiated by st.text_input() below.
    pending_query = st.session_state.pop("_pending_kg_query", None)
    auto_search = pending_query is not None
    if pending_query is not None:
        st.session_state["kg_query"] = pending_query

    # Search bar
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "ابحث عن كيان",
            placeholder="مثال: القدس، محمود درويش، التطريز، يافا",
            label_visibility="collapsed",
            key="kg_query",
        )
    with col2:
        st.markdown("<div style='height:2.4rem'></div>", unsafe_allow_html=True)
        search_clicked = st.button("بحث", use_container_width=True, type="primary")

    # Quick search chips — real buttons, not decorative <span> markup (which
    # has no Python-side click handler at all: clicking one previously did
    # nothing because Streamlit had no way to know it was clicked).
    # keyed by list position, not the entity name — sample_entities can
    # contain duplicate names (distinct unlinked entities sharing a
    # canonical name, e.g. two separate "القدس" nodes), which collided.
    CHIPS_PER_ROW = 5
    for row_start in range(0, len(sample_entities), CHIPS_PER_ROW):
        row_entities = list(enumerate(sample_entities))[row_start:row_start + CHIPS_PER_ROW]
        chip_cols = st.columns(CHIPS_PER_ROW)
        for col, (idx, entity) in zip(chip_cols, row_entities):
            with col:
                if st.button(entity, key=f"kg_chip_{idx}", use_container_width=True):
                    st.session_state["_pending_kg_query"] = entity
                    st.rerun()

    # Stats bar
    st.markdown(
        f"""<div style='display:flex;gap:2rem;margin-bottom:1rem;padding:0.75rem 1rem;"""
        f"""background:#FAF8F5;border-radius:8px;border:1px solid #E8E2D9;'>"""
        f"""<span style='color:#8B8580;font-size:0.85rem;'>إجمالي الكيانات: <b style='color:#2C2C2C;'>{stats['total_entities']:,}</b></span>"""
        f"""<span style='color:#8B8580;font-size:0.85rem;'>العلاقات: <b style='color:#2C2C2C;'>{stats['total_relations']:,}</b></span>"""
        f"""<span style='color:#8B8580;font-size:0.85rem;'>مرتبطة بـ Wikidata: <b style='color:#2C2C2C;'>{stats['linked_to_wikidata']:,}</b></span>"""
        f"""</div>""",
        unsafe_allow_html=True,
    )

    # Perform search
    if query and (search_clicked or auto_search):
        graph_data = graph_search_fn(query)
        center = graph_data["center"]
        neighbors = graph_data["neighbors"]

        col1, col2 = st.columns([3, 2])

        with col1:
            fig = create_kg_plotly_graph(graph_data)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, theme=None)

        with col2:
            # Entity info panel
            wikidata_str = ""
            if center.get("wikidata"):
                wikidata_str = f"""<div class='metadata-item'>
                    Wikidata: <a href='https://www.wikidata.org/wiki/{center['wikidata']}' target='_blank' style='color:#009736;'>{center['wikidata']}</a>
                </div>"""

            # wikidata_str is inlined onto the same line as the next div
            # rather than on its own line — when it's "" (no Wikidata QID),
            # a lone blank line inside this HTML block trips CommonMark's
            # "blank line ends an HTML block" rule, and everything after it
            # renders as escaped literal text instead of markup.
            st.markdown(
                f"""<div class='entity-info-panel'>
                    <div style='font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;color:#2C2C2C;'>
                        {center['name']}
                    </div>
                    <div class='metadata-item'>
                        نوع الكيان: <strong>{center['type']}</strong>
                    </div>
                    {wikidata_str}<div class='metadata-item'>
                        عدد العلاقات: <strong>{len(neighbors)}</strong>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Relations list
            st.markdown("<div style='font-size:0.95rem;font-weight:600;margin:0.75rem 0 0.5rem;color:#2C2C2C;'>العلاقات</div>", unsafe_allow_html=True)
            if not neighbors:
                st.caption("لا توجد علاقات مسجلة لهذا الكيان.")
            for neighbor in neighbors:
                arrow = "→" if neighbor["direction"] == "out" else "←"
                st.markdown(
                    f"""<div style='padding:0.4rem 0.6rem;border-bottom:1px solid #F0EDE8;font-size:0.88rem;'>
                        <span style='color:#8B8580;'>{center['name']}</span>
                        <span style='color:#C5A55A;margin:0 0.3rem;'>{arrow} {neighbor['relation']}</span>
                        <span style='color:#009736;font-weight:500;'>{neighbor['name']}</span>
                        <span style='color:#8B8580;font-size:0.75rem;margin-right:0.3rem;'>({neighbor['type']})</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
    elif query and not search_clicked:
        pass
    else:
        # Empty state
        st.markdown(
            """<div class='empty-state'>
                <div class='empty-state-icon'>⚛️</div>
                <div class='empty-state-text'>ابحث عن كيان لاستكشاف الرسم المعرفي</div>
                <div class='empty-state-hint'>جرب: القدس، محمود درويش، التطريز، يافا</div>
            </div>""",
            unsafe_allow_html=True,
        )
