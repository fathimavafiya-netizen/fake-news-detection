"""
Sentiment analysis using NLTK VADER.
"""

import nltk

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyze sentiment of text using VADER.

    Returns:
        dict with 'compound', 'positive', 'negative', 'neutral',
        'label', 'interpretation'
    """
    if not text or not isinstance(text, str):
        return {
            'compound': 0.0, 'positive': 0.0, 'negative': 0.0,
            'neutral': 1.0, 'label': 'Neutral',
            'interpretation': 'No text to analyze.'
        }

    scores = _analyzer.polarity_scores(text)

    compound = scores['compound']
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'

    # Subjectivity heuristic: high positive + negative = subjective
    subjectivity = scores['pos'] + scores['neg']
    if subjectivity > 0.5:
        subj_text = "highly subjective"
    elif subjectivity > 0.25:
        subj_text = "moderately subjective"
    else:
        subj_text = "mostly objective"

    tone = "positive" if compound > 0 else ("negative" if compound < 0 else "neutral")
    interpretation = (
        f"This text has a {tone} tone (compound: {compound:.3f}) "
        f"and is {subj_text}. "
        f"Fake news tends to use more emotionally charged language."
    )

    return {
        'compound': round(compound, 4),
        'positive': round(scores['pos'], 4),
        'negative': round(scores['neg'], 4),
        'neutral': round(scores['neu'], 4),
        'label': label,
        'interpretation': interpretation
    }
