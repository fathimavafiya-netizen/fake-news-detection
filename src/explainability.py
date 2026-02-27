"""
Explainability module: feature importance, attention visualization, bias detection.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def explain_baseline_prediction(text, predictor, top_n=10):
    """
    Explain a baseline (TF-IDF + NB) prediction.

    Returns:
        dict with 'prediction', 'top_words' (list of (word, score, direction))
    """
    result = predictor.predict(text)
    contributions = predictor.get_feature_weights(text, top_n=top_n)

    top_words = []
    for word, score in contributions:
        direction = 'towards Real' if score > 0 else 'towards Fake'
        top_words.append({
            'word': word,
            'score': round(abs(score), 4),
            'direction': direction,
            'raw_score': round(score, 4)
        })

    return {
        'prediction': result,
        'top_words': top_words,
        'explanation_type': 'tfidf_nb_coefficients'
    }


def explain_bert_prediction(text, predictor, top_n=15):
    """
    Explain a BERT prediction using attention weights.

    Returns:
        dict with 'prediction', 'token_attention' (list of (token, weight))
    """
    result = predictor.predict(text)

    try:
        token_attn = predictor.get_attention_weights(text)
        # Filter out special tokens and padding
        filtered = [(tok, w) for tok, w in token_attn
                     if tok not in ('[CLS]', '[SEP]', '[PAD]')]
        # Normalize weights
        if filtered:
            max_w = max(w for _, w in filtered)
            if max_w > 0:
                filtered = [(tok, w / max_w) for tok, w in filtered]
        # Top N
        filtered.sort(key=lambda x: x[1], reverse=True)
        token_attention = [{'token': tok, 'weight': round(w, 4)}
                           for tok, w in filtered[:top_n]]
    except Exception:
        token_attention = []

    return {
        'prediction': result,
        'token_attention': token_attention,
        'explanation_type': 'bert_attention'
    }


def generate_word_importance_html(words_with_scores, max_words=20):
    """
    Generate HTML with highlighted words (color intensity = importance).

    Args:
        words_with_scores: list of dicts with 'word', 'score', 'direction'
    Returns:
        HTML string
    """
    html_parts = []
    for item in words_with_scores[:max_words]:
        word = item['word']
        score = item['score']
        direction = item.get('direction', '')

        # Color: red for fake, green for real
        if 'Fake' in direction:
            r, g, b = 255, int(255 * (1 - score)), int(255 * (1 - score))
        else:
            r, g, b = int(255 * (1 - score)), 255, int(255 * (1 - score))

        opacity = min(0.3 + score * 0.7, 1.0)
        html_parts.append(
            f'<span style="background-color: rgba({r},{g},{b},{opacity}); '
            f'padding: 2px 6px; margin: 2px; border-radius: 4px; '
            f'display: inline-block; font-size: 14px;">'
            f'{word} ({score:.3f})</span>'
        )

    return ' '.join(html_parts)


def plot_word_importance(words_with_scores, save_path=None, top_n=15):
    """
    Plot horizontal bar chart of word importance.

    Returns:
        matplotlib figure
    """
    items = words_with_scores[:top_n]
    if not items:
        return None

    words = [item['word'] for item in items]
    scores = [item['raw_score'] for item in items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(items) * 0.4)))
    colors = ['#f44336' if s < 0 else '#4CAF50' for s in scores]
    y_pos = range(len(items))

    ax.barh(y_pos, scores, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.set_xlabel('Contribution Score')
    ax.set_title('Word Importance (Green=Real, Red=Fake)')
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ---- Bias Detection (keyword-based) ----

BIAS_LEXICONS = {
    'left': [
        'progressive', 'liberal', 'equality', 'diversity', 'inclusion',
        'social justice', 'climate change', 'regulation', 'universal',
        'welfare', 'immigrant rights', 'reform', 'systemic', 'marginalized',
        'affordable', 'green', 'renewable', 'healthcare', 'workers rights'
    ],
    'right': [
        'conservative', 'traditional', 'freedom', 'liberty', 'patriot',
        'deregulation', 'free market', 'border security', 'law and order',
        'second amendment', 'tax cuts', 'small government', 'national security',
        'family values', 'fiscal responsibility', 'sovereignty', 'military'
    ]
}


def detect_bias(text):
    """
    Detect political bias using keyword matching.

    Returns:
        dict with 'score' (-1=left, +1=right), 'label', 'left_count', 'right_count'
    """
    if not text or not isinstance(text, str):
        return {'score': 0.0, 'label': 'Neutral', 'left_count': 0, 'right_count': 0}

    text_lower = text.lower()

    left_count = sum(1 for phrase in BIAS_LEXICONS['left'] if phrase in text_lower)
    right_count = sum(1 for phrase in BIAS_LEXICONS['right'] if phrase in text_lower)

    total = left_count + right_count
    if total == 0:
        score = 0.0
        label = 'Neutral'
    else:
        score = (right_count - left_count) / total  # -1 to +1
        if score < -0.3:
            label = 'Left-leaning'
        elif score > 0.3:
            label = 'Right-leaning'
        else:
            label = 'Center'

    return {
        'score': round(score, 3),
        'label': label,
        'left_count': left_count,
        'right_count': right_count
    }


# ---- Source Credibility ----

TRUSTED_SOURCES = {
    'reuters.com': 1.0, 'apnews.com': 1.0, 'bbc.com': 0.95, 'bbc.co.uk': 0.95,
    'nytimes.com': 0.9, 'washingtonpost.com': 0.9, 'theguardian.com': 0.9,
    'npr.org': 0.9, 'pbs.org': 0.9, 'cnn.com': 0.8, 'abcnews.go.com': 0.8,
    'cbsnews.com': 0.8, 'nbcnews.com': 0.8, 'usatoday.com': 0.8,
}

UNRELIABLE_SOURCES = {
    'infowars.com': 0.1, 'naturalnews.com': 0.1, 'beforeitsnews.com': 0.1,
    'worldnewsdailyreport.com': 0.05, 'theonion.com': 0.0,
}

import re

def assess_source_credibility(text):
    """
    Extract domains from text and assess source credibility.

    Returns:
        dict with 'domains_found', 'credibility_score', 'assessment'
    """
    urls = re.findall(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
    domains = list(set(urls))

    if not domains:
        return {
            'domains_found': [],
            'credibility_score': 0.5,
            'assessment': 'No source domains detected in text.'
        }

    scores = []
    details = []
    for domain in domains:
        domain = domain.lower()
        if domain in TRUSTED_SOURCES:
            score = TRUSTED_SOURCES[domain]
            details.append(f"{domain}: trusted ({score:.1f})")
        elif domain in UNRELIABLE_SOURCES:
            score = UNRELIABLE_SOURCES[domain]
            details.append(f"{domain}: unreliable ({score:.1f})")
        else:
            score = 0.5
            details.append(f"{domain}: unknown (0.5)")
        scores.append(score)

    avg_score = np.mean(scores)
    if avg_score >= 0.8:
        assessment = "Sources appear highly credible."
    elif avg_score >= 0.5:
        assessment = "Source credibility is uncertain."
    else:
        assessment = "Sources have low credibility ratings."

    return {
        'domains_found': details,
        'credibility_score': round(float(avg_score), 2),
        'assessment': assessment
    }
