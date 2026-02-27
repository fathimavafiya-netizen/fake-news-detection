"""
Text preprocessing pipeline for Fake News Detection.
Provides 3 levels of cleaning: baseline (aggressive), LSTM (moderate), BERT (minimal).
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import pandas as pd


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    resources = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)


ensure_nltk_data()

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Negation words to keep for LSTM (they carry semantic meaning)
NEGATION_WORDS = {'not', 'no', 'nor', 'neither', 'never', 'none',
                  "n't", 'cannot', "can't", "won't", "don't", "doesn't",
                  "didn't", "isn't", "aren't", "wasn't", "weren't"}


def remove_html_tags(text):
    """Strip HTML tags from text."""
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_emails(text):
    """Remove email addresses from text."""
    return re.sub(r'\S+@\S+\.\S+', '', text)


def remove_special_characters(text):
    """Remove special characters, keeping only alphanumeric and spaces."""
    return re.sub(r'[^a-zA-Z\s]', '', text)


def remove_extra_whitespace(text):
    """Collapse multiple whitespace into single space."""
    return re.sub(r'\s+', ' ', text).strip()


def clean_text_baseline(text):
    """
    Aggressive cleaning for TF-IDF + Naive Bayes.
    Removes HTML, URLs, special chars, stopwords, and applies lemmatization.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = text.lower()
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)

    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]

    return ' '.join(tokens)


def clean_text_lstm(text):
    """
    Moderate cleaning for LSTM.
    Keeps punctuation and negation words for sequence learning.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,!?\'-]', '', text)
    text = remove_extra_whitespace(text)

    # Light stopword removal: keep negation words
    tokens = word_tokenize(text)
    stop_minus_negation = STOP_WORDS - NEGATION_WORDS
    tokens = [t for t in tokens if t not in stop_minus_negation or t in NEGATION_WORDS]

    return ' '.join(tokens)


def clean_text_bert(text):
    """
    Minimal cleaning for BERT/DistilBERT.
    Only removes URLs and emails; BERT tokenizer handles the rest.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_extra_whitespace(text)

    return text


def combine_title_text(title, text, separator=" "):
    """Merge headline and body text."""
    title = str(title).strip() if pd.notna(title) else ""
    text = str(text).strip() if pd.notna(text) else ""

    if title and text:
        return f"{title}{separator}{text}"
    return title or text


def remove_duplicates(df, subset='text'):
    """Remove exact duplicate articles."""
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
    removed = initial_len - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows.")
    return df


def preprocess_dataset(df, method='baseline'):
    """
    Apply preprocessing to the entire dataset.

    Args:
        df: DataFrame with 'title', 'text', 'label' columns
        method: 'baseline', 'lstm', or 'bert'
    Returns:
        DataFrame with added 'clean_text' column
    """
    clean_fn = {
        'baseline': clean_text_baseline,
        'lstm': clean_text_lstm,
        'bert': clean_text_bert,
    }.get(method, clean_text_baseline)

    df = df.copy()
    df['combined'] = df.apply(
        lambda row: combine_title_text(row.get('title', ''), row.get('text', '')),
        axis=1
    )
    df['clean_text'] = df['combined'].apply(clean_fn)
    # Drop rows where clean text is empty
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    return df
