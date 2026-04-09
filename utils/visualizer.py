"""
utils/visualizer.py
-------------------
All chart/visualization functions for the Streamlit UI.
Uses Plotly for interactive charts and Matplotlib for word clouds.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ── Color palette ──────────────────────────────
COLORS = {
    "primary": "#FF6B35",
    "secondary": "#2EC4B6",
    "accent": "#FFBF00",
    "dark": "#1A1A2E",
    "light": "#F8F9FA",
    "success": "#06D6A0",
    "danger": "#EF233C",
}
PALETTE = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
           COLORS["success"], "#9B5DE5", "#F15BB5"]


def plot_similarity_scores(results: list, top_n: int = 10) -> go.Figure:
    """Bar chart of top recipe similarity scores."""
    top = results[:top_n]
    titles = [r["title"][:35] + "..." if len(r["title"]) > 35 else r["title"] for r in top]
    scores = [r["score"] * 100 for r in top]

    fig = go.Figure(go.Bar(
        x=scores,
        y=titles,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[[0, '#1A1A2E'], [0.5, COLORS["secondary"]], [1, COLORS["primary"]]],
            showscale=False,
        ),
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside',
    ))
    fig.update_layout(
        title="Recipe Match Scores",
        xaxis_title="Similarity (%)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
        height=400,
        margin=dict(l=10, r=80, t=50, b=40),
    )
    return fig


def plot_pos_distribution(pos_counts: dict) -> go.Figure:
    """Pie chart of Part-of-Speech tag distribution."""
    pos_labels = {
        "NN": "Nouns", "VB": "Verbs", "JJ": "Adjectives",
        "RB": "Adverbs", "DT": "Determiners", "IN": "Prepositions",
        "CC": "Conjunctions", "CD": "Numbers", "PR": "Pronouns",
    }
    labels, values = [], []
    for k, v in pos_counts.items():
        label = pos_labels.get(k[:2], k)
        if label not in labels:
            labels.append(label)
            values.append(v)
        else:
            idx = labels.index(label)
            values[idx] += v

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=PALETTE,
    ))
    fig.update_layout(
        title="Part-of-Speech Distribution",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
        height=350,
        showlegend=True,
    )
    return fig


def plot_word_frequency(top_words: list) -> go.Figure:
    """Horizontal bar for word frequencies."""
    if not top_words:
        return go.Figure()
    words = [w for w, _ in top_words]
    freqs = [f for _, f in top_words]

    fig = go.Figure(go.Bar(
        x=freqs,
        y=words,
        orientation='h',
        marker_color=COLORS["secondary"],
        text=freqs,
        textposition='outside',
    ))
    fig.update_layout(
        title="Top Keywords (Word-Level Analysis)",
        xaxis_title="Frequency",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
        height=350,
    )
    return fig


def plot_tfidf_keywords(query_keywords: list) -> go.Figure:
    """Bar chart showing TF-IDF weights for query keywords."""
    if not query_keywords:
        return go.Figure()
    words = [w for w, _ in query_keywords]
    weights = [round(v, 4) for _, v in query_keywords]

    fig = go.Figure(go.Bar(
        x=words,
        y=weights,
        marker=dict(
            color=weights,
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
        ),
        text=[f"{w:.3f}" for w in weights],
        textposition='outside',
    ))
    fig.update_layout(
        title="TF-IDF Keyword Weights (Semantic Analysis)",
        xaxis_title="Keywords",
        yaxis_title="TF-IDF Weight",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
        height=320,
    )
    return fig


def plot_nlp_pipeline_flow() -> go.Figure:
    """Sankey-style flow showing NLP pipeline stages."""
    stages = ["Raw Text", "Tokenization", "Stopword Removal", "Lemmatization",
              "TF-IDF Vectorization", "Cosine Similarity", "Ranked Results"]

    fig = go.Figure()
    x_positions = np.linspace(0.05, 0.95, len(stages))
    colors_seq = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  COLORS["success"], "#9B5DE5", "#F15BB5", COLORS["primary"]]

    # Draw arrows
    for i in range(len(stages) - 1):
        fig.add_annotation(
            x=x_positions[i + 1] - 0.065, y=0.5,
            ax=-70, ay=0,
            xref="paper", yref="paper",
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#ccc",
        )

    # Draw stage nodes
    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        fig.add_shape(
            type="rect",
            x0=x - 0.06, x1=x + 0.06, y0=0.3, y1=0.7,
            xref="paper", yref="paper",
            fillcolor=colors_seq[i], opacity=0.9,
            line=dict(width=0),
        )
        fig.add_annotation(
            x=x, y=0.5, text=f"<b>{stage}</b>",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="white", size=10),
            align="center",
        )

    fig.update_layout(
        title="NLP Processing Pipeline",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(l=0, r=0, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def recipe_card_html(title: str, ingredients: str, instructions: str, score: float = None) -> str:
    """Returns styled HTML for a recipe card."""
    score_badge = f'<span class="score-badge">🎯 {score*100:.1f}% match</span>' if score else ""
    ing_preview = ingredients[:200] + "..." if len(ingredients) > 200 else ingredients
    instr_preview = instructions[:300] + "..." if len(instructions) > 300 else instructions

    return f"""
    <div class="recipe-card">
        <div class="recipe-title">{title} {score_badge}</div>
        <div class="recipe-section">
            <span class="section-label">🥬 Ingredients</span>
            <p class="recipe-text">{ing_preview}</p>
        </div>
        <div class="recipe-section">
            <span class="section-label">👨‍🍳 Instructions</span>
            <p class="recipe-text">{instr_preview}</p>
        </div>
    </div>
    """
