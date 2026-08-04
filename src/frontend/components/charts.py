"""Plotly chart components with consistent Palestinian palette."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent palette
PALETTE = {
    "green": "#009736",
    "red": "#CE1126",
    "gold": "#C5A55A",
    "olive": "#6B8E23",
    "dark_olive": "#556B2F",
    "black": "#1a1a1a",
    "warm_gray": "#8B8580",
    "light_gray": "#D4CFC8",
    "ivory": "#FAF8F5",
}

COLOR_SEQ = ["#009736", "#CE1126", "#C5A55A", "#6B8E23", "#556B2F", "#8B8580", "#1a1a1a"]

BASE_LAYOUT = dict(
    # Explicit light template — without it, st.plotly_chart()'s default
    # theme="streamlit" auto-styles figures from the *visiting browser's*
    # OS dark-mode preference, not from styles/theme.css, and would render
    # near-white axis/legend text invisible against these light cards. Kept
    # even though every st.plotly_chart() call also passes theme=None, as
    # a second line of defense.
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Noto Sans Arabic, Segoe UI, sans-serif", size=12, color="#2C2C2C"),
    margin=dict(l=10, r=10, t=40, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
    ),
)


def _axis_style(showgrid=True):
    return dict(
        gridcolor="#E8E2D9",
        gridwidth=1,
        showgrid=showgrid,
        zeroline=False,
        tickfont=dict(size=11),
    )


def create_source_bar_chart(data):
    """Bar chart of documents by source."""
    sources = [d["source"] for d in data]
    counts = [d["count"] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sources, y=counts,
        marker_color=[COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(len(sources))],
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="%{x}<br>عدد الوثائق: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis=_axis_style(showgrid=False),
        yaxis={**_axis_style(), "title": {"text": "عدد الوثائق", "font": {"size": 11}}},
        title={"text": "توزيع الوثائق حسب المصدر", "font": {"size": 14, "color": "#2C2C2C"}},
        height=320,
    )
    return fig


def create_language_donut(data):
    """Donut chart of documents by language."""
    labels = [d["language"] for d in data]
    values = [d["count"] for d in data]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=[PALETTE["green"], PALETTE["gold"], PALETTE["warm_gray"]][: len(labels)],
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{label}<br>%{value} وثيقة (%{percent})<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        showlegend=False,
        height=320,
        title={"text": "الوثائق حسب اللغة", "font": {"size": 14, "color": "#2C2C2C"}},
    )
    return fig


def create_topic_bar(data):
    """Horizontal bar chart for topic distribution."""
    topics = [d["topic"] for d in data]
    counts = [d["count"] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=topics, x=counts,
        orientation="h",
        marker_color=[COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(len(topics))],
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="%{y}<br>عدد الوثائق: %{x}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**_axis_style(), "title": {"text": "عدد الوثائق", "font": {"size": 11}}},
        yaxis={**_axis_style(showgrid=False)},
        height=340,
        title={"text": "توزيع المواضيع الثقافية", "font": {"size": 14, "color": "#2C2C2C"}},
    )
    return fig


def create_entity_type_chart(data):
    """Bar chart for KG entity types."""
    labels = [d["label"] for d in data]
    counts = [d["count"] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=counts,
        marker_color=PALETTE["olive"],
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="%{x}<br>عدد الكيانات: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**_axis_style(showgrid=False), "tickangle": -30},
        yaxis={**_axis_style(), "title": {"text": "العدد", "font": {"size": 11}}},
        height=320,
        title={"text": "أنواع كيانات الرسم المعرفي", "font": {"size": 14, "color": "#2C2C2C"}},
    )
    return fig


def create_topic_scatter(documents, selected_topics=None, highlight_topic=None):
    """Scatter plot for topic map (UMAP-style visualization)."""
    df_filtered = documents
    if selected_topics is not None:
        df_filtered = [d for d in documents if d["topic_id"] in selected_topics]

    # Group by topic for traces
    topic_groups = {}
    for doc in df_filtered:
        tid = doc["topic_id"]
        if tid not in topic_groups:
            topic_groups[tid] = {"x": [], "y": [], "name": doc["topic_name"], "color": doc["color"]}
        topic_groups[tid]["x"].append(doc["x"])
        topic_groups[tid]["y"].append(doc["y"])

    fig = go.Figure()

    for tid, group in sorted(topic_groups.items(), key=lambda kv: str(kv[0])):
        is_highlight = highlight_topic is not None and tid == highlight_topic
        opacity = 1.0 if is_highlight or highlight_topic is None else 0.25
        size = 10 if is_highlight else 6

        fig.add_trace(go.Scattergl(
            x=group["x"], y=group["y"],
            mode="markers",
            name=group["name"],
            marker=dict(
                size=size,
                color=group["color"],
                opacity=opacity,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "الموضوع: " + group["name"] + "<br>"
                "المصدر: %{customdata[0]}<br>"
                "اللغة: %{customdata[1]}<extra></extra>"
            ),
            text=[doc["title"] for doc in df_filtered if doc["topic_id"] == tid],
            customdata=[[doc["source"], doc["language"]] for doc in df_filtered if doc["topic_id"] == tid],
        ))

    # This chart can have 20+ topics (one legend entry each). BASE_LAYOUT's
    # shared legend is a horizontal strip pinned at a fixed y offset below
    # the plot — fine for 2-4 entries, but with this many it wraps across
    # several rows and overlaps the x-axis title/ticks beneath it (Plotly
    # doesn't reflow the axis to make room for a legend of unknown height).
    # A vertical legend avoids that specific collision but, with this many
    # entries, ends up taller than the plot itself and gets clipped instead
    # — trading one overlap bug for another. Simplest robust fix: no legend
    # here at all. With real BERTopic output, topic "names" are dense
    # keyword strings, not label-sized text anyway (a known, documented
    # limitation — see F1 in ROADMAP.md); the hover tooltip already shows
    # each point's topic, and the "اختر موضوعًا للتفصيل" selector below
    # drives the real per-topic detail panel.
    fig.update_layout(
        **{**BASE_LAYOUT, "showlegend": False, "margin": dict(l=10, r=20, t=40, b=40)},
        xaxis={**_axis_style(), "title": {"text": "البعد 1", "font": {"size": 11}}},
        yaxis={**_axis_style(), "title": {"text": "البعد 2", "font": {"size": 11}}},
        height=560,
        title={"text": "خريطة المواضيع", "font": {"size": 14, "color": "#2C2C2C"}},
        dragmode="lasso",
    )
    return fig


def create_timeline_bar(data):
    """Timeline bar chart: documents by decade."""
    decades = [d["decade"] for d in data]
    counts = [d["count"] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=decades, y=counts,
        marker_color=PALETTE["green"],
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="العقد: %{x}<br>عدد الوثائق: %{y}<extra></extra>",
    ))

    # Add a vertical line for 1948 — uses add_shape()/add_annotation()
    # directly rather than the add_vline() convenience wrapper, because
    # add_vline() unconditionally runs its shape through annotation-position
    # math that averages x0/x1 as numbers, which crashes on a categorical
    # (string) x-axis like these decade labels, even with no annotation text.
    if "1940" in decades:
        fig.add_shape(
            type="line", x0="1940", x1="1940", y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(width=2, dash="dash", color=PALETTE["red"]),
        )
        fig.add_annotation(
            x="1940", y=1, yref="paper", yanchor="bottom",
            text="1948 النكبة", showarrow=False,
            font=dict(size=10, color=PALETTE["red"]),
        )

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**_axis_style(showgrid=False), "title": {"text": "العقد", "font": {"size": 11}}},
        yaxis={**_axis_style(), "title": {"text": "عدد الوثائق", "font": {"size": 11}}},
        height=380,
        title={"text": "توزيع الوثائق عبر الزمن", "font": {"size": 14, "color": "#2C2C2C"}},
    )
    return fig


def create_term_frequency_line(term_name: str, freq_data, extra_terms=None):
    """Line chart showing term frequency over time."""
    fig = go.Figure()

    years = [d["year"] for d in freq_data]
    freqs = [d["freq"] for d in freq_data]

    fig.add_trace(go.Scatter(
        x=years, y=freqs,
        mode="lines+markers",
        name=term_name,
        line=dict(color=PALETTE["green"], width=3),
        marker=dict(size=8, color=PALETTE["green"]),
        fill="tozeroy",
        fillcolor="rgba(0,151,54,0.08)",
        hovertemplate=f"{term_name}<br>العقد: %{{x}}<br>التكرار: %{{y}}<extra></extra>",
    ))

    if extra_terms:
        for term, tdata in extra_terms:
            fig.add_trace(go.Scatter(
                x=[d["year"] for d in tdata],
                y=[d["freq"] for d in tdata],
                mode="lines+markers",
                name=term,
                line=dict(width=2, dash="dot"),
                marker=dict(size=5),
                hovertemplate=f"{term}<br>العقد: %{{x}}<br>التكرار: %{{y}}<extra></extra>",
            ))

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**_axis_style(), "title": {"text": "العقد", "font": {"size": 11}}},
        yaxis={**_axis_style(), "title": {"text": "التكرار", "font": {"size": 11}}},
        height=360,
        title={"text": f"تكرار المصطلح عبر الزمن: {term_name}", "font": {"size": 14, "color": "#2C2C2C"}},
    )
    return fig


def create_bias_gauge(dimension_name: str, value: float, left_label: str, right_label: str):
    """Create a horizontal gauge chart for a bias dimension.

    The dimension name is passed as the Indicator's own `title`, not a
    separate figure-level `update_layout(title=...)`. Those two don't
    coordinate — a figure title doesn't reserve space from the indicator's
    own number/gauge layout, so with a short figure (~220px) the percentage
    text ended up squeezed into the same vertical space as the arc and
    visually clipped by it. Title+number+gauge all belonging to the same
    Indicator trace is what makes Plotly lay them out without overlap.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": dimension_name, "font": {"size": 13, "color": "#2C2C2C"}},
        number={
            "font": {"size": 30, "color": "#2C2C2C"},
            "suffix": "%",
        },
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#D4CFC8"},
            "bar": {"color": PALETTE["green"], "thickness": 0.7},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E8E2D9",
            "steps": [
                {"range": [0, 50], "color": "rgba(206,17,38,0.08)"},
                {"range": [50, 75], "color": "rgba(197,165,90,0.12)"},
                {"range": [75, 100], "color": "rgba(0,151,54,0.08)"},
            ],
        },
    ))

    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=20, r=20, t=10, b=10)},
        height=320,
    )
    return fig


def create_bias_source_comparison(data, categories=None):
    """Stacked bar chart comparing content categories across sources.

    Args:
        data: list of dicts, each with a "source" key plus one numeric key
            per category (percentages).
        categories: list of (key, label) pairs describing which keys to plot
            and their Arabic display label, in stacking order. Defaults to
            the original 4-category demo taxonomy for backward compatibility,
            but real usage always passes the categories actually present in
            the data (the real ContentCategory taxonomy has up to 17 values,
            not a fixed 4).
    """
    if categories is None:
        categories = [
            ("cultural_pct", "الثقافة"),
            ("conflict_pct", "الصراع"),
            ("historical_pct", "التاريخ"),
            ("religious_pct", "الدين"),
        ]

    sources = [d["source"] for d in data]
    colors = [COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(len(categories))]

    fig = go.Figure()
    for (key, label), color in zip(categories, colors):
        fig.add_trace(go.Bar(
            name=label,
            x=sources,
            y=[d.get(key, 0) for d in data],
            marker_color=color,
            hovertemplate=f"{label}: %{{y}}%<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="stack",
        xaxis={**_axis_style(showgrid=False), "tickangle": -20},
        yaxis={**_axis_style(), "title": {"text": "النسبة المئوية", "font": {"size": 11}}},
        height=360,
        title={"text": "مقارنة توزيع المحتوى حسب المصدر", "font": {"size": 14, "color": "#2C2C2C"}},
    )
    return fig


def create_bias_dimension_radar(data):
    """Radar chart showing bias dimensions."""
    dims = [d["dimension"] for d in data]
    vals = [d["value"] for d in data]

    fig = go.Figure(go.Scatterpolar(
        r=vals,
        theta=dims,
        fill="toself",
        fillcolor="rgba(0,151,54,0.15)",
        line=dict(color=PALETTE["green"], width=2),
        marker=dict(size=6, color=PALETTE["green"]),
        hovertemplate="%{theta}: %{r}%<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        polar=dict(
            radialaxis=dict(
                range=[0, 100],
                tickfont=dict(size=10),
                gridcolor="#E8E2D9",
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                rotation=-90,
                direction="clockwise",
            ),
        ),
        height=380,
        showlegend=False,
        title={"text": "أبعاد التمثيل المحتوى", "font": {"size": 14, "color": "#2C2C2C"}},
    )
    return fig


def create_kg_plotly_graph(graph_data):
    """Create a Plotly network graph for KG visualization.

    This component is isolated so the visualization library (Plotly/PyVis/NetworkX)
    can be swapped later.
    """
    import math

    center = graph_data["center"]
    neighbors = graph_data["neighbors"]

    # Calculate positions in a radial layout
    n = max(len(neighbors), 1)
    center_x, center_y = 0, 0

    edge_x = []
    edge_y = []
    node_x = [center_x]
    node_y = [center_y]
    node_colors = [PALETTE["green"]]
    node_sizes = [40]
    node_labels = [center["name"]]
    node_types = [center["type"]]

    type_color_map = {
        "PERSON": PALETTE["gold"],
        "LOCATION": PALETTE["green"],
        "HERITAGE_PLACE": PALETTE["olive"],
        "HERITAGE_FOOD": PALETTE["red"],
        "HERITAGE_CRAFT": PALETTE["dark_olive"],
        "HERITAGE_CLOTHING": PALETTE["gold"],
        "ORGANIZATION": PALETTE["warm_gray"],
        "MISC": PALETTE["light_gray"],
    }

    for i, neighbor in enumerate(neighbors):
        angle = 2 * math.pi * i / n - math.pi / 2
        radius = 2.0
        nx = center_x + radius * math.cos(angle)
        ny = center_y + radius * math.sin(angle)

        # Edge
        if neighbor["direction"] == "out":
            edge_x.extend([center_x, nx, None])
            edge_y.extend([center_y, ny, None])
        else:
            edge_x.extend([nx, center_x, None])
            edge_y.extend([ny, center_y, None])

        node_x.append(nx)
        node_y.append(ny)
        node_labels.append(neighbor["name"])
        node_types.append(neighbor["type"])
        node_sizes.append(25)
        node_colors.append(type_color_map.get(neighbor["type"], PALETTE["warm_gray"]))

    # Build edges trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#D4CFC8"),
        hoverinfo="none",
        mode="lines",
    )

    # Build nodes trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, color="#2C2C2C"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="white"),
            opacity=0.9,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "النوع: %{customdata}<extra></extra>"
        ),
        customdata=node_types,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**_axis_style(showgrid=False), "showticklabels": False, "zeroline": False},
        yaxis={**_axis_style(showgrid=False), "showticklabels": False, "zeroline": False},
        height=480,
        showlegend=False,
        dragmode="pan",
        title={
            "text": f"الرسم المعرفي: {center['name']}",
            "font": {"size": 14, "color": "#2C2C2C"},
        },
    )
    return fig
