"""
nlp/preprocessor.py
-------------------
NLP Pipeline: Word-Level, Syntax, Semantic & Discourse Analysis
"""

import re
import string
import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
def download_nltk_data():
    resources = [
        'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
        'maxent_ne_chunker', 'words', 'punkt_tab', 'averaged_perceptron_tagger_eng'
    ]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_data()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Graceful fallback if NLTK corpora not yet downloaded
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
        'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'both', 'either',
        'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very', 's', 't', 'just',
        'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
        'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
        'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'in', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'out', 'on', 'off', 'over', 'under', 'again', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'because', 'as', 'until', 'while',
    }

# ─────────────────────────────────────────────
#  LEVEL 1: WORD-LEVEL ANALYSIS
# ─────────────────────────────────────────────

def safe_tokenize(text: str) -> list:
    """Tokenize with fallback to regex if punkt not available."""
    try:
        return word_tokenize(text.lower())
    except Exception:
        return re.findall(r"\b\w+\b", text.lower())


def safe_sent_tokenize(text: str) -> list:
    """Sentence tokenize with fallback."""
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def safe_lemmatize(word: str) -> str:
    try:
        return lemmatizer.lemmatize(word)
    except Exception:
        return word


def safe_pos_tag(tokens: list) -> list:
    try:
        return pos_tag(tokens)
    except Exception:
        # Naive fallback: label anything ending in common suffixes
        result = []
        for t in tokens:
            if t.endswith(("ing", "ed", "ate", "ize")):
                result.append((t, "VB"))
            elif t.endswith(("ly",)):
                result.append((t, "RB"))
            elif t.endswith(("ful", "ous", "ish", "ive", "al")):
                result.append((t, "JJ"))
            else:
                result.append((t, "NN"))
        return result


def word_level_analysis(text: str) -> dict:
    """
    Performs tokenization, stopword removal, stemming, lemmatization.
    This is the LEXICAL layer of NLP.
    """
    # Tokenize
    tokens = safe_tokenize(text.lower())

    # Remove punctuation
    tokens_clean = [t for t in tokens if t not in string.punctuation]

    # Stopword removal
    tokens_no_stop = [t for t in tokens_clean if t not in STOP_WORDS]

    # Stemming (aggressive normalization)
    stemmed = [stemmer.stem(t) for t in tokens_no_stop]

    # Lemmatization (dictionary-based normalization)
    lemmatized = [safe_lemmatize(t) for t in tokens_no_stop]

    # Frequency distribution
    freq_dist = {}
    for token in tokens_no_stop:
        freq_dist[token] = freq_dist.get(token, 0) + 1

    top_words = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "original_tokens": tokens_clean,
        "tokens_no_stopwords": tokens_no_stop,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
        "vocabulary_size": len(set(tokens_no_stop)),
        "total_tokens": len(tokens_clean),
        "top_words": top_words,
        "stopwords_removed": len(tokens_clean) - len(tokens_no_stop),
    }


# ─────────────────────────────────────────────
#  LEVEL 2: SYNTAX ANALYSIS
# ─────────────────────────────────────────────

def syntax_analysis(text: str) -> dict:
    """
    Performs Part-of-Speech tagging and Named Entity Recognition.
    This is the STRUCTURAL / GRAMMATICAL layer of NLP.
    """
    sentences = safe_sent_tokenize(text)
    pos_results = []
    all_tags = []

    for sent in sentences[:3]:  # Limit for performance
        tokens = safe_tokenize(sent)
        tagged = safe_pos_tag(tokens)
        pos_results.append({"sentence": sent, "tags": tagged})
        all_tags.extend(tagged)

    # Count POS types
    pos_counts = {}
    for word, tag in all_tags:
        short_tag = tag[:2]
        pos_counts[short_tag] = pos_counts.get(short_tag, 0) + 1

    # Extract nouns and verbs (important for recipe understanding)
    nouns = [w for w, t in all_tags if t.startswith('NN')]
    verbs = [w for w, t in all_tags if t.startswith('VB')]
    adjectives = [w for w, t in all_tags if t.startswith('JJ')]

    return {
        "sentences": sentences,
        "pos_tagged": pos_results,
        "pos_counts": pos_counts,
        "nouns": list(set(nouns))[:10],
        "verbs": list(set(verbs))[:10],
        "adjectives": list(set(adjectives))[:10],
        "sentence_count": len(sentences),
    }


# ─────────────────────────────────────────────
#  LEVEL 3: SEMANTIC & DISCOURSE ANALYSIS
# ─────────────────────────────────────────────

def semantic_analysis(query: str, corpus_texts: list, corpus_titles: list) -> dict:
    """
    TF-IDF based semantic similarity — finds meaning beyond exact keywords.
    This is the MEANING / SEMANTIC layer of NLP.
    
    Search Technique: TF-IDF + Cosine Similarity
    """
    if not corpus_texts:
        return {"error": "Empty corpus"}

    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigrams + Bigrams
        stop_words='english',
        min_df=1
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
        query_vec = vectorizer.transform([query])

        # Cosine similarity between query and all recipes
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "index": int(idx),
                    "title": corpus_titles[idx],
                    "score": float(similarities[idx]),
                    "percentage": f"{similarities[idx]*100:.1f}%"
                })

        # Feature importance (what keywords drove the match)
        feature_names = vectorizer.get_feature_names_out()
        query_tfidf = query_vec.toarray()[0]
        top_feature_indices = np.argsort(query_tfidf)[::-1][:10]
        query_keywords = [
            (feature_names[i], float(query_tfidf[i]))
            for i in top_feature_indices
            if query_tfidf[i] > 0
        ]

        return {
            "results": results,
            "query_keywords": query_keywords,
            "total_recipes_searched": len(corpus_texts),
            "matches_found": len(results),
        }

    except Exception as e:
        return {"error": str(e), "results": []}


def clean_ingredients(raw: str) -> str:
    """Clean raw ingredient strings from the dataset."""
    if not isinstance(raw, str):
        return ""
    # Remove list artifacts like ['1 cup', '2 tbsp']
    raw = re.sub(r"[\[\]']", "", raw)
    raw = re.sub(r"\d+\s*(Ã½|Â¼|Â¾|â|â|½|¼|¾)", " ", raw)
    raw = raw.replace("\\n", " ").replace("\\t", " ")
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def preprocess_for_search(text: str) -> str:
    """Full NLP pipeline for indexing: clean → tokenize → lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = safe_tokenize(text)
    tokens = [safe_lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)
