"""Palestinian Cultural Knowledge Platform — Streamlit Frontend.

مِنَصَّةُ المَعْرِفَةِ الثَّقَافِيَّةِ الفِلَسْطِينِيَّةِ

Run: uv run streamlit run src/frontend/app.py

Architecture:
  app.py                    — entry point, navigation, page routing
  components/header.py      — hero, accent line, footer
  components/cards.py       — KPI cards, section headers, badges
  components/charts.py      — all Plotly chart factories
  components/views/         — one file per tab: overview, topic_map, timeline, bias, kg_explorer, ask
  services/backend.py       — real pipeline data (data_loaders.py, graph_store.py, bias/temporal
                               reports, topic_model output, the RAG pipeline for Ask)
  mock/demo_data.py         — kept for reference/offline demoing, not used by default
  styles/theme.css          — CSS theme (Palestinian + Islamic visual identity)
"""

import sys
import os

# This file's own directory, so "components"/"services"/"mock" resolve as
# top-level packages, and the repository root two levels up, so
# services/backend.py can import "src.xxx" pipeline modules.
FRONTEND_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(FRONTEND_ROOT))
if FRONTEND_ROOT not in sys.path:
    sys.path.insert(0, FRONTEND_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st

# ── Services ──────────────────────────────────────────────
from services.backend import (
    get_overview_data,
    get_topic_data,
    get_timeline_data,
    get_bias_data,
    get_kg_data,
    search_knowledge_graph,
    ask_question,
)

# ── Components ────────────────────────────────────────────
from components.header import render_accent_line, render_hero, render_footer
from components.views.overview import render as render_overview
from components.views.topic_map import render as render_topic_map
from components.views.timeline import render as render_timeline
from components.views.bias import render as render_bias
from components.views.kg_explorer import render as render_kg
from components.views.ask import render as render_ask

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="منصة المعرفة الثقافية الفلسطينية",
    page_icon="\U0001F33F",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load CSS ──────────────────────────────────────────────
with open(os.path.join(FRONTEND_ROOT, "styles", "theme.css"), "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
render_accent_line()
render_hero()

# ── Navigation ────────────────────────────────────────────
# st.tabs() labels render as plain/lightly-markdown text, not arbitrary HTML —
# a raw "<span class='tab-en'>" (as originally drafted) shows up as literal
# text instead of a styled subtitle, so each tab is just its Arabic name.
TAB_NAMES = [
    "نظرة عامة",
    "خريطة المواضيع",
    "الخط الزمني",
    "مقياس الانحياز",
    "مستكشف الرسم المعرفي",
    "اسأل المنصة",
]

selected = st.tabs(TAB_NAMES)

# ── Tab 0: Overview ──────────────────────────────────────
with selected[0]:
    data = get_overview_data()
    render_overview(data)

# ── Tab 1: Topic Map ─────────────────────────────────────
with selected[1]:
    data = get_topic_data()
    render_topic_map(data)

# ── Tab 2: Timeline ──────────────────────────────────────
with selected[2]:
    data = get_timeline_data()
    render_timeline(data)

# ── Tab 3: Bias Meter ────────────────────────────────────
with selected[3]:
    data = get_bias_data()
    render_bias(data)

# ── Tab 4: KG Explorer ────────────────────────────────────
with selected[4]:
    data = get_kg_data()
    render_kg(data, search_knowledge_graph)

# ── Tab 5: Ask ───────────────────────────────────────────
with selected[5]:
    render_ask(ask_question)

# ── Footer ────────────────────────────────────────────────
render_footer()
