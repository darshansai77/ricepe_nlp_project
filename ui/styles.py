"""
ui/styles.py
------------
All custom CSS styles for the Streamlit application.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --primary: #FF6B35;
    --secondary: #2EC4B6;
    --accent: #FFBF00;
    --dark: #1A1A2E;
    --card-bg: #ffffff;
    --text: #2d3748;
    --subtext: #718096;
    --border: #e2e8f0;
    --success: #06D6A0;
}

/* ── Global ── */
.stApp {
    font-family: 'DM Sans', sans-serif;
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 30%, #a1c4fd 70%, #c2e9fb 100%);
    background-attachment: fixed;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, var(--dark) 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(26,26,46,0.3);
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,107,53,0.15) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: white;
    margin: 0;
    letter-spacing: -1px;
    position: relative;
    z-index: 1;
}
.app-title span { color: var(--primary); }
.app-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.5rem;
    position: relative;
    z-index: 1;
}

/* ── Search Box ── */
.search-container {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    margin-bottom: 1.5rem;
    border: 2px solid transparent;
    transition: border-color 0.3s;
}
.search-container:hover { border-color: var(--primary); }

/* ── Recipe Cards ── */
.recipe-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-left: 5px solid var(--primary);
    transition: transform 0.2s, box-shadow 0.2s;
}
.recipe-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}
.recipe-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--dark);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.score-badge {
    background: linear-gradient(135deg, var(--primary), #ff9a5c);
    color: white;
    font-size: 0.75rem;
    font-family: 'DM Sans', sans-serif;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 600;
}
.section-label {
    display: inline-block;
    background: #f0fffe;
    color: var(--secondary);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.recipe-text {
    font-size: 0.92rem;
    color: var(--subtext);
    line-height: 1.6;
    margin: 0.3rem 0 0.8rem;
}
.recipe-section { margin-bottom: 0.5rem; }

/* ── NLP Analysis Cards ── */
.nlp-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
}
.nlp-card-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--subtext);
    margin-bottom: 0.6rem;
}
.token-chip {
    display: inline-block;
    background: #f0f4f8;
    color: var(--text);
    font-size: 0.78rem;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    margin: 0.15rem;
    font-family: 'DM Mono', monospace;
}
.token-chip.highlight { background: #fff3e0; color: var(--primary); font-weight: 600; }
.pos-tag {
    display: inline-block;
    padding: 0.15rem 0.4rem;
    border-radius: 6px;
    font-size: 0.7rem;
    margin: 0.1rem;
    font-weight: 600;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.stat-item {
    background: white;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    flex: 1;
    min-width: 120px;
}
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
}
.stat-label {
    font-size: 0.72rem;
    color: var(--subtext);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px;
    padding: 0.3rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* ── Sidebar ── */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--dark) 0%, #16213e 100%) !important;
}
.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: white;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.sidebar-info {
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
    border-left: 3px solid var(--secondary);
}
.sidebar-info p {
    color: rgba(255,255,255,0.8);
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.5;
}
.sidebar-tag {
    display: inline-block;
    background: rgba(255,107,53,0.2);
    color: #ffb8a0;
    font-size: 0.72rem;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    margin: 0.15rem;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: white;
    border-radius: 10px;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton>button {
    background: linear-gradient(135deg, var(--primary) 0%, #ff9a5c 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(255,107,53,0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255,107,53,0.4);
}

/* ── Info/Alert boxes ── */
.info-box {
    background: linear-gradient(135deg, #e8f5ff, #f0fffe);
    border: 1px solid #b3e0ff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: var(--text);
}

/* ── No results ── */
.no-results {
    text-align: center;
    padding: 3rem;
    color: var(--subtext);
    font-size: 1.1rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem;
    color: var(--subtext);
    font-size: 0.85rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
}
</style>
"""


def get_pos_color(tag: str) -> str:
    """Return color for POS tag chip."""
    colors = {
        "NN": "#fff3e0", "VB": "#e8f5e9", "JJ": "#e3f2fd",
        "RB": "#fce4ec", "DT": "#f3e5f5", "IN": "#e0f7fa",
        "CC": "#fff8e1", "CD": "#e8eaf6",
    }
    return colors.get(tag[:2], "#f5f5f5")


def get_pos_text_color(tag: str) -> str:
    text_colors = {
        "NN": "#e65100", "VB": "#1b5e20", "JJ": "#0d47a1",
        "RB": "#880e4f", "DT": "#4a148c", "IN": "#006064",
        "CC": "#f57f17", "CD": "#1a237e",
    }
    return text_colors.get(tag[:2], "#555")
