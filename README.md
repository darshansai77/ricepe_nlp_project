# 🍽️ Intelligent Recipe Recommendation System using NLP

> A full NLP pipeline project — Word-Level Analysis, Syntax Analysis, Semantic Search — built with Python, NLTK, scikit-learn, and Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

---

## 🧠 NLP Concepts Demonstrated

| Layer | Technique | Purpose |
|-------|-----------|---------|
| **Word-Level** | Tokenization, Stopword Removal, Stemming, Lemmatization | Normalize text |
| **Syntax** | POS Tagging (Nouns, Verbs, Adjectives) | Understand structure |
| **Semantic** | TF-IDF + Cosine Similarity | Find meaning-based matches |
| **Discourse** | Full-context recipe indexing | Holistic understanding |

---

## 🔍 Search Technique

**TF-IDF (Term Frequency × Inverse Document Frequency)** vectorizes all recipes into numerical vectors. **Cosine Similarity** measures how close your query vector is to each recipe vector — finding semantically relevant recipes even when exact words don't match.

---

## 📁 Project Structure

```
recipe_nlp_project/
├── app.py                    ← Streamlit entry point
├── requirements.txt
│
├── nlp/
│   ├── preprocessor.py       ← All NLP functions (word, syntax, semantic)
│   └── __init__.py
│
├── data/
│   ├── loader.py             ← Dataset loading + preprocessing
│   └── __init__.py
│
├── utils/
│   ├── visualizer.py         ← Plotly charts and recipe cards
│   └── __init__.py
│
└── ui/
    ├── styles.py             ← Custom CSS
    └── __init__.py
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/recipe-nlp.git
cd recipe-nlp

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add Kaggle dataset
# Download from: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images
# Place CSV in: data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv

# 4. Run the app
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** [Kaggle - Food Ingredients and Recipe Dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images/data)
- **Fallback:** Built-in sample of 20 diverse recipes (auto-loaded if CSV not present)
- **Columns used:** Title, Ingredients, Instructions

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path:** `app.py`
5. Click **Deploy** ✅

---

## 🛠️ Tech Stack

| Tool | Use |
|------|-----|
| **Streamlit** | Web UI |
| **NLTK** | Tokenization, POS tagging, Lemmatization |
| **scikit-learn** | TF-IDF Vectorizer, Cosine Similarity |
| **Plotly** | Interactive charts |
| **Pandas / NumPy** | Data processing |

---

## 📸 Features

- 🔍 **Semantic Search** — finds recipes by meaning, not just exact keywords
- 🔤 **Word Analysis Tab** — visualize tokenization, stemming, lemmatization
- 🧬 **Syntax Tab** — POS tag distribution, noun/verb extraction
- ⚙️ **Pipeline Overview** — explains every NLP step visually
- 📊 **Match Score Charts** — shows similarity scores for all results

---

## 👨‍💻 Author

Built as an NLP project demonstrating applied natural language processing on real-world food data.
