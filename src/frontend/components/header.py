"""Header component with hero section and Palestinian identity."""

import streamlit as st


def render_accent_line():
    """Render the Palestinian flag accent line at the top."""
    st.markdown(
        "<div class='palestinian-accent-line'></div>",
        unsafe_allow_html=True,
    )


def render_hero():
    """Render the main hero section with title and description."""
    st.markdown(
        """
        <div class="hero-section">
            <h1 class="hero-title">مِنَصَّةُ المَعْرِفَةِ الثَّقَافِيَّةِ الفِلَسْطِينِيَّةِ</h1>
            <p class="hero-subtitle">
                استكشاف المعرفة والثقافة الفلسطينية من خلال الذكاء الاصطناعي ومعالجة اللغة الطبيعية.
            </p>
            <p class="hero-desc">
                <span class="tech-badge">Corpus</span>
                <span class="tech-badge">NLP</span>
                <span class="tech-badge">Knowledge Graph</span>
                <span class="tech-badge">RAG</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_olive_decoration():
    """Render subtle olive branch decoration."""
    st.markdown(
        "<div class='olive-decoration'>✉ ✉ ✉</div>",
        unsafe_allow_html=True,
    )


def render_footer():
    """Render the platform footer."""
    st.markdown(
        """
        <div class="platform-footer">
            <div class="footer-olive">✉ ✉ ✉</div>
            <div>منصة المعرفة الثقافية الفلسطينية &mdash; preserving Palestinian cultural knowledge through AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
