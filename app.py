"""
app.py
------
Intelligent Recipe Recommendation System using NLP
Main Streamlit Application Entry Point

NLP Layers Demonstrated:
  1. Word-Level Analysis  → Tokenization, Stemming, Lemmatization
  2. Syntax Analysis      → POS Tagging, Named Entity Recognition
  3. Semantic Analysis    → TF-IDF + Cosine Similarity (Search)

Run: streamlit run app.py
"""

import streamlit as st

# ── Page Config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title="Recipe NLP",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──
import pandas as pd

from data.loader import load_dataset, get_recipe_by_index
from nlp.preprocessor import (
    word_level_analysis,
    syntax_analysis,
    semantic_analysis,
    preprocess_for_search,
)
from utils.visualizer import (
    plot_similarity_scores,
    plot_pos_distribution,
    plot_word_frequency,
    plot_tfidf_keywords,
    plot_nlp_pipeline_flow,
    recipe_card_html,
)
from ui.styles import CUSTOM_CSS, get_pos_color, get_pos_text_color

# ── Apply CSS ──
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════
def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🍽️ Recipe NLP</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-info">
            <p><b style="color:#2EC4B6">NLP Layers:</b></p>
            <span class="sidebar-tag">Word-Level</span>
            <span class="sidebar-tag">Syntax</span>
            <span class="sidebar-tag">Semantic</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-info">
            <p><b style="color:#FFBF00">Search Technique:</b><br>
            TF-IDF + Cosine Similarity finds recipes that match the <em>meaning</em> of your query, not just exact words.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-info">
            <p><b style="color:#FF6B35">Dataset:</b><br>
            Kaggle Food Ingredients & Recipe Dataset<br>
            (loads sample data if CSV not present)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style="color:rgba(255,255,255,0.5); font-size:0.8rem; text-align:center;">
            📦 {len(df)} recipes loaded<br>
            🔍 TF-IDF Vectorization<br>
            📐 Cosine Similarity Search
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        # Quick ingredient search chips
        st.markdown('<p style="color:rgba(255,255,255,0.7); font-size:0.85rem; font-weight:600;">Try searching for:</p>', unsafe_allow_html=True)
        quick_searches = ["chicken garlic", "chocolate dessert", "pasta italian",
                          "vegetarian healthy", "soup comfort", "egg breakfast"]
        return st.selectbox("Quick examples:", [""] + quick_searches, label_visibility="collapsed")


# ════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════
def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🍳 Intelligent <span>Recipe</span> Recommender</div>
        <div class="app-subtitle">
            Natural Language Processing · TF-IDF Search · Word Analysis · Syntax Parsing · Semantic Matching
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
#  WORD LEVEL ANALYSIS TAB
# ════════════════════════════════════════════
def render_word_analysis(query: str):
    if not query.strip():
        st.info("Enter a search query above to see Word-Level Analysis.")
        return

    analysis = word_level_analysis(query)

    st.markdown("### 🔤 Level 1: Word-Level Analysis")
    st.markdown("""
    <div class="info-box">
    <b>What this does:</b> Breaks raw text into tokens, removes filler words (stopwords),
    and normalizes words via <b>Stemming</b> (chops endings) and <b>Lemmatization</b> (dictionary lookup).
    This is the foundation of all NLP pipelines.
    </div>
    """, unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tokens", analysis["total_tokens"])
    col2.metric("After Stopword Removal", len(analysis["tokens_no_stopwords"]))
    col3.metric("Stopwords Removed", analysis["stopwords_removed"])
    col4.metric("Unique Vocabulary", analysis["vocabulary_size"])

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🔵 Original Tokens:**")
        chips = " ".join(
            f'<span class="token-chip">{t}</span>' for t in analysis["original_tokens"][:20]
        )
        st.markdown(f'<div class="nlp-card">{chips}</div>', unsafe_allow_html=True)

        st.markdown("**🟢 After Stopword Removal:**")
        chips2 = " ".join(
            f'<span class="token-chip highlight">{t}</span>' for t in analysis["tokens_no_stopwords"][:20]
        )
        st.markdown(f'<div class="nlp-card">{chips2}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("**🔴 Stemmed:**")
        stems = " ".join(
            f'<span class="token-chip" style="background:#ffe0e0;color:#c53030">{t}</span>'
            for t in analysis["stemmed"][:20]
        )
        st.markdown(f'<div class="nlp-card">{stems}</div>', unsafe_allow_html=True)

        st.markdown("**🟡 Lemmatized:**")
        lemmas = " ".join(
            f'<span class="token-chip" style="background:#fffde0;color:#b7791f">{t}</span>'
            for t in analysis["lemmatized"][:20]
        )
        st.markdown(f'<div class="nlp-card">{lemmas}</div>', unsafe_allow_html=True)

    # Frequency chart
    if analysis["top_words"]:
        st.plotly_chart(plot_word_frequency(analysis["top_words"]), use_container_width=True)


# ════════════════════════════════════════════
#  SYNTAX ANALYSIS TAB
# ════════════════════════════════════════════
def render_syntax_analysis(query: str):
    if not query.strip():
        st.info("Enter a search query above to see Syntax Analysis.")
        return

    analysis = syntax_analysis(query)

    st.markdown("### 🧬 Level 2: Syntax Analysis")
    st.markdown("""
    <div class="info-box">
    <b>What this does:</b> Assigns grammatical roles to each word using
    <b>Part-of-Speech (POS) Tagging</b>. Identifies Nouns (ingredients), Verbs (cook actions),
    Adjectives (descriptors). This structural layer helps the system understand <em>what type</em>
    of word each token is.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentences", analysis["sentence_count"])
    col2.metric("Nouns (Ingredients)", len(analysis["nouns"]))
    col3.metric("Verbs (Actions)", len(analysis["verbs"]))

    # POS Tagged view
    if analysis["pos_tagged"]:
        st.markdown("**📌 POS Tagged Tokens:**")
        for sent_data in analysis["pos_tagged"][:2]:
            sent_html = ""
            for word, tag in sent_data["tags"][:25]:
                bg = get_pos_color(tag)
                tc = get_pos_text_color(tag)
                sent_html += f'<span class="pos-tag" style="background:{bg};color:{tc}">{word}<sub style="font-size:0.6em;opacity:0.7">{tag}</sub></span>'
            st.markdown(f'<div class="nlp-card">{sent_html}</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # POS distribution chart
        if analysis["pos_counts"]:
            st.plotly_chart(plot_pos_distribution(analysis["pos_counts"]), use_container_width=True)

    with col_b:
        if analysis["nouns"]:
            st.markdown("**🟠 Nouns (Key Ingredients):**")
            noun_html = " ".join(
                f'<span class="token-chip highlight">{n}</span>' for n in analysis["nouns"]
            )
            st.markdown(f'<div class="nlp-card">{noun_html}</div>', unsafe_allow_html=True)

        if analysis["verbs"]:
            st.markdown("**🟢 Verbs (Cooking Actions):**")
            verb_html = " ".join(
                f'<span class="token-chip" style="background:#e8f5e9;color:#276749">{v}</span>'
                for v in analysis["verbs"]
            )
            st.markdown(f'<div class="nlp-card">{verb_html}</div>', unsafe_allow_html=True)

        if analysis["adjectives"]:
            st.markdown("**🔵 Adjectives (Descriptors):**")
            adj_html = " ".join(
                f'<span class="token-chip" style="background:#e3f2fd;color:#1565c0">{a}</span>'
                for a in analysis["adjectives"]
            )
            st.markdown(f'<div class="nlp-card">{adj_html}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
#  SEMANTIC SEARCH TAB
# ════════════════════════════════════════════
def render_semantic_search(query: str, df: pd.DataFrame, top_n: int):
    if not query.strip():
        st.info("Enter a search query above to see Semantic Results.")
        return

    st.markdown("### 🔮 Level 3: Semantic & Discourse Analysis")
    st.markdown("""
    <div class="info-box">
    <b>What this does:</b> Uses <b>TF-IDF vectorization</b> to convert all recipes into numerical vectors,
    then computes <b>Cosine Similarity</b> between your query and every recipe.
    Finds recipes that are <em>semantically similar</em> — even if they don't share exact words.
    This is the core <b>search technique</b> of the system.
    </div>
    """, unsafe_allow_html=True)

    # Run semantic search
    with st.spinner("🔍 Running TF-IDF + Cosine Similarity search..."):
        corpus_texts = df["search_corpus"].tolist()
        corpus_titles = df["Title"].tolist()
        sem_result = semantic_analysis(query, corpus_texts, corpus_titles)

    if "error" in sem_result:
        st.error(f"Search error: {sem_result['error']}")
        return

    results = sem_result.get("results", [])

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Recipes Searched", sem_result.get("total_recipes_searched", 0))
    col2.metric("Matches Found", sem_result.get("matches_found", 0))
    top_score = results[0]["score"] * 100 if results else 0
    col3.metric("Best Match Score", f"{top_score:.1f}%")

    # TF-IDF keyword weights
    if sem_result.get("query_keywords"):
        st.plotly_chart(plot_tfidf_keywords(sem_result["query_keywords"]), use_container_width=True)

    # Similarity scores bar chart
    if results:
        st.plotly_chart(plot_similarity_scores(results, top_n=min(top_n, 10)), use_container_width=True)

        # Show recipe cards
        st.markdown("### 🍽️ Top Recommended Recipes")
        for res in results[:top_n]:
            recipe = get_recipe_by_index(df, res["index"])
            st.markdown(
                recipe_card_html(
                    recipe["title"],
                    recipe["ingredients"],
                    recipe["instructions"],
                    score=res["score"]
                ),
                unsafe_allow_html=True
            )
    else:
        st.markdown('<div class="no-results">😕 No matching recipes found. Try different keywords.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════
#  NLP PIPELINE OVERVIEW TAB
# ════════════════════════════════════════════
def render_pipeline_overview():
    st.markdown("### ⚙️ How the NLP Pipeline Works")
    st.plotly_chart(plot_nlp_pipeline_flow(), use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🔤 Level 1: Word-Level Analysis (Lexical)
        The first step in any NLP system.

        | Step | What it does |
        |------|-------------|
        **Tokenization** | Splits text into individual words |
        **Stopword Removal** | Removes "the", "a", "is" etc. |
        **Stemming** | `cooking` → `cook` (rule-based) |
        **Lemmatization** | `better` → `good` (dictionary) |

        ---

        #### 🧬 Level 2: Syntax Analysis (Structural)
        Understands grammatical structure.

        | Tag | Meaning | Example |
        |-----|---------|---------|
        NN | Noun | chicken, pasta |
        VB | Verb | bake, mix, cook |
        JJ | Adjective | crispy, fresh |
        RB | Adverb | quickly, gently |
        """)

    with col2:
        st.markdown("""
        #### 🔮 Level 3: Semantic Analysis (Meaning)
        The core search algorithm.

        **TF-IDF** *(Term Frequency × Inverse Document Frequency)*

        - **TF**: How often a word appears in a recipe
        - **IDF**: How rare the word is across all recipes
        - Rare-but-relevant words get higher weight

        **Cosine Similarity**
        - Converts text → numerical vectors
        - Measures angle between query vector & recipe vectors
        - Score 1.0 = perfect match, 0.0 = no relation

        ---

        #### 🔁 Discourse Analysis
        The system reads *full recipe context* —
        not just the title but also ingredients and
        instructions — giving a holistic semantic match.

        **Why this beats keyword search:**
        A query for *"healthy vegetarian dinner"* finds
        recipes by *meaning*, even if they say
        *"nutritious meatless meal"* instead.
        """)

    st.markdown("---")
    st.markdown("""
    #### 📁 Project File Structure
    ```
    recipe_nlp_project/
    ├── app.py                    ← Main Streamlit entry point
    ├── requirements.txt          ← Dependencies
    │
    ├── nlp/
    │   ├── __init__.py
    │   └── preprocessor.py       ← Word-Level, Syntax, Semantic NLP
    │
    ├── data/
    │   ├── __init__.py
    │   └── loader.py             ← Dataset loading & preprocessing
    │
    ├── utils/
    │   ├── __init__.py
    │   └── visualizer.py         ← All charts and visualizations
    │
    ├── ui/
    │   ├── __init__.py
    │   └── styles.py             ← Custom CSS themes
    │
    └── data/
        └── (place Kaggle CSV here)
    ```
    """)


# ════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════
def main():
    # Load data
    with st.spinner("Loading recipe dataset..."):
        df = load_dataset()

    # Sidebar
    quick_search = render_sidebar(df)

    # Header
    render_header()

    # ── Search Input ──
    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        default_q = quick_search if quick_search else ""
        query = st.text_input(
            "🔍 Search for recipes by ingredients, cuisine, or dish name:",
            value=default_q,
            placeholder="e.g. 'spicy chicken with garlic and lemon'",
            label_visibility="visible",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Search 🚀", use_container_width=True)

    top_n = st.slider("Number of results:", min_value=3, max_value=15, value=5)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🍽️ Recipe Results",
        "🔤 Word Analysis",
        "🧬 Syntax Analysis",
        "⚙️ How It Works"
    ])

    with tab1:
        render_semantic_search(query, df, top_n)

    with tab2:
        render_word_analysis(query)

    with tab3:
        render_syntax_analysis(query)

    with tab4:
        render_pipeline_overview()

    # ── Footer ──
    st.markdown("""
    <div class="footer">
        Built with ❤️ using Python · Streamlit · NLTK · scikit-learn · Plotly<br>
        <b>NLP Project</b> — Word-Level · Syntax · Semantic Analysis · TF-IDF Search
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
