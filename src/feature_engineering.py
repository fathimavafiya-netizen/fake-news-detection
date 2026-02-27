"""
Feature engineering: TF-IDF vectorizer and LSTM tokenizer builders.
"""

import os
import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import MODELS_DIR


def build_tfidf_vectorizer(texts, max_features=10000, ngram_range=(1, 2),
                           min_df=5, max_df=0.7):
    """
    Build and fit a TF-IDF vectorizer on the given texts.

    Returns:
        vectorizer: fitted TfidfVectorizer
        tfidf_matrix: sparse matrix of TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        dtype=np.float32
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer, tfidf_matrix


def save_vectorizer(vectorizer, model_dir=None):
    """Save TF-IDF vectorizer to pickle."""
    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, 'baseline')
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {path}")


def load_vectorizer(model_dir=None):
    """Load TF-IDF vectorizer from pickle."""
    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, 'baseline')
    path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_lstm_tokenizer(texts, vocab_size=20000, oov_token='<OOV>'):
    """
    Build a Keras-compatible tokenizer for LSTM.

    Returns:
        word_index: dict mapping words to integer indices
        tokenizer_config: dict with tokenizer settings (JSON-serializable)
    """
    from collections import Counter

    # Build vocabulary from training texts
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # Keep top vocab_size words
    most_common = word_counts.most_common(vocab_size - 1)  # -1 for OOV
    word_index = {oov_token: 1}  # 0 reserved for padding
    for idx, (word, _) in enumerate(most_common, start=2):
        word_index[word] = idx

    config = {
        'vocab_size': len(word_index) + 1,  # +1 for padding index 0
        'oov_token': oov_token,
        'actual_vocab': len(word_index)
    }

    print(f"LSTM tokenizer: {config['actual_vocab']} words (+ padding)")
    return word_index, config


def texts_to_sequences(texts, word_index, max_length=500):
    """
    Convert texts to padded integer sequences using word_index.

    Returns:
        numpy array of shape (num_texts, max_length)
    """
    sequences = []
    oov_idx = word_index.get('<OOV>', 1)

    for text in texts:
        seq = [word_index.get(word, oov_idx) for word in text.split()]
        sequences.append(seq)

    # Pad/truncate
    padded = np.zeros((len(sequences), max_length), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]

    return padded


def save_lstm_tokenizer(word_index, config, model_dir=None):
    """Save LSTM tokenizer to JSON."""
    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, 'lstm')
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'word_index.json'), 'w') as f:
        json.dump(word_index, f)
    with open(os.path.join(model_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"LSTM tokenizer saved to {model_dir}")


def load_lstm_tokenizer(model_dir=None):
    """Load LSTM tokenizer from JSON."""
    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, 'lstm')
    with open(os.path.join(model_dir, 'word_index.json'), 'r') as f:
        word_index = json.load(f)
    with open(os.path.join(model_dir, 'tokenizer_config.json'), 'r') as f:
        config = json.load(f)
    return word_index, config
