"""
Dashboard (Track G2) — Streamlit, per ROADMAP.md's own recommendation
("a working prototype in two hours," free HuggingFace Spaces hosting).

Panels: Overview, Topic Map, Timeline, Bias Meter, KG Explorer, Ask (live RAG).
Every panel except "Ask" reads from files already produced by earlier tracks
(src/frontend/data_loaders.py) — no live Postgres connection needed to view
prior analysis results. Only "Ask" needs Postgres+Ollama, and is wrapped so
a DB/Ollama outage degrades that one panel instead of crashing the app.

Run:
    uv run streamlit run src/frontend/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import plotly.express as px
import streamlit as st

from src.frontend.data_loaders import (
    DEFAULT_BIAS_REPORT_PATH,
    DEFAULT_KG_GRAPH_PATH,
    DEFAULT_TEMPORAL_REPORT_PATH,
    load_corpus_overview,
    load_json_report,
    load_kg_graph_summary,
    load_topic_distribution,
)

st.set_page_config(page_title="Palestinian Cultural Knowledge Platform", layout="wide")
st.title("Palestinian Cultural Knowledge Platform")

TAB_NAMES = ["Overview", "Topic Map", "Timeline", "Bias Meter", "KG Explorer", "Ask"]
overview_tab, topic_tab, timeline_tab, bias_tab, kg_tab, ask_tab = st.tabs(TAB_NAMES)

with overview_tab:
    st.header("Corpus Overview")
    overview = load_corpus_overview()
    if overview["total_documents"] == 0:
        st.info("No corpus files found yet — run a collector first (see README.md's Data Collection section).")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total documents", overview["total_documents"])
        with col1:
            st.plotly_chart(
                px.bar(x=list(overview["by_source"].keys()), y=list(overview["by_source"].values()), title="By source"),
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                px.pie(names=list(overview["by_language"].keys()), values=list(overview["by_language"].values()), title="By language"),
                width="stretch",
            )

    kg_summary = load_kg_graph_summary(DEFAULT_KG_GRAPH_PATH)
    if kg_summary:
        st.subheader("Knowledge Graph")
        col1, col2 = st.columns(2)
        col1.metric("Entities", kg_summary["num_nodes"])
        col2.metric("Relations", kg_summary["num_edges"])
    else:
        st.caption("Knowledge graph not built yet — run scripts/build_kg_graph.py.")

with topic_tab:
    st.header("Topic Map")
    topics_files = list(Path("data/processed").glob("*.topics.jsonl"))
    distribution = load_topic_distribution(topics_files)
    if not distribution:
        st.info(
            "No topic assignments found — run `uv run python scripts/run_topic_model.py` first "
            "(requires the pgvector chunk index to be built via scripts/build_index.py)."
        )
    else:
        st.plotly_chart(
            px.bar(
                x=list(distribution.values()),
                y=list(distribution.keys()),
                orientation="h",
                title="Documents per topic (top 20)",
            ),
            width="stretch",
        )

with timeline_tab:
    st.header("Timeline")
    report = load_json_report(DEFAULT_TEMPORAL_REPORT_PATH)
    if not report:
        st.info("No temporal analysis found — run `uv run python scripts/run_temporal_analysis.py` first.")
    else:
        decades = report["decade_distribution"]
        st.plotly_chart(
            px.bar(x=[str(d) for d in decades.keys()], y=list(decades.values()), title="Documents by decade"),
            width="stretch",
        )
        st.caption(
            f"{report['documents_with_metadata_decade']} documents dated from real metadata, "
            f"{report['documents_with_content_estimated_decade']} estimated from years mentioned in the text "
            f"(see ROADMAP.md Track F4 for why)."
        )
        st.subheader("Term frequency by decade (per 1,000 words)")
        term_freq = report["term_frequency_per_1000_words_by_decade"]
        selected_term = st.selectbox("Term", list(term_freq.keys()))
        if selected_term:
            series = term_freq[selected_term]
            sorted_decades = sorted(series.keys(), key=int)
            st.plotly_chart(
                px.line(x=sorted_decades, y=[series[d] for d in sorted_decades], title=selected_term, markers=True),
                width="stretch",
            )

with bias_tab:
    st.header("Bias Meter")
    report = load_json_report(DEFAULT_BIAS_REPORT_PATH)
    if not report:
        st.info("No bias measurement found — run `uv run python scripts/run_bias_measurement.py` first.")
    else:
        st.subheader("Category distribution by source")
        for source_id, distribution in report["category_distribution_by_source"].items():
            st.write(f"**{source_id}**")
            st.plotly_chart(
                px.bar(x=list(distribution.keys()), y=list(distribution.values()), title=source_id),
                width="stretch",
            )
        st.subheader("Cross-source divergence (total variation distance, 0=identical, 1=disjoint)")
        st.json(report["category_distribution_divergence"])
        st.subheader("WEAT embedding association test")
        st.metric("Effect size", report["weat"]["effect_size"])
        st.caption(report["weat"]["interpretation"])
        if report.get("llm_framing_probe_by_source"):
            st.subheader("LLM framing probe by source")
            st.json(report["llm_framing_probe_by_source"])

with kg_tab:
    st.header("Knowledge Graph Explorer")
    if not DEFAULT_KG_GRAPH_PATH.exists():
        st.info("No knowledge graph found — run scripts/build_kg_graph.py first.")
    else:
        from src.knowlegde_graph.graph_store import find_entities_by_name, load_graph, neighbors_of

        graph = load_graph(DEFAULT_KG_GRAPH_PATH)
        query = st.text_input("Search entity by name (Arabic or partial match)")
        if query:
            matches = find_entities_by_name(graph, query)[:20]
            if not matches:
                st.warning("No matching entities found.")
            for entity_id in matches:
                node = graph.nodes[entity_id]
                with st.expander(f"{node.get('canonical_name')} ({node.get('type')})"):
                    if node.get("wikidata_qid"):
                        st.write(f"Wikidata: [{node['wikidata_qid']}](https://www.wikidata.org/wiki/{node['wikidata_qid']})")
                    neighbors = neighbors_of(graph, entity_id)
                    if neighbors:
                        st.table(neighbors)
                    else:
                        st.caption("No outgoing relations recorded.")

with ask_tab:
    st.header("Ask the Corpus")
    st.caption("Live RAG query — requires Postgres (docker compose up -d) and Ollama running.")
    question = st.text_input("Your question (Arabic or English)")
    if st.button("Ask") and question:
        try:
            from src.rag.config import RagConfig
            from src.rag.db import get_connection
            from src.rag.embedder import Embedder
            from src.rag.generator import OllamaGenerator
            from src.rag.pipeline import RAGPipeline
            from src.rag.retriever import Retriever

            config = RagConfig.load()
            embedder = Embedder(config.embedding)
            conn = get_connection()
            try:
                pipeline = RAGPipeline(Retriever(conn, embedder, config), OllamaGenerator(config.generation), config)
                answer = pipeline.ask(question)
                st.write(answer.text)
                for citation in answer.citations:
                    st.caption(f"[{citation.index}] {citation.title or '(untitled)'} — {citation.source_url or 'n/a'}")
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - show a friendly message, don't crash the dashboard
            st.error(f"Couldn't answer that right now: {exc}")
