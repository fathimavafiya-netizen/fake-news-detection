"""
Baseline model training: TF-IDF + Multinomial Naive Bayes.
Run: python -m src.train_baseline
"""

import os
import sys
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

from src.utils import load_raw_dataset, split_dataset, load_split, MODELS_DIR, DATA_PROCESSED
from src.preprocessing import preprocess_dataset, remove_duplicates
from src.feature_engineering import build_tfidf_vectorizer, save_vectorizer
from src.evaluate import full_evaluation


def train_baseline():
    """Train TF-IDF + Naive Bayes baseline model."""

    model_dir = os.path.join(MODELS_DIR, 'baseline')
    os.makedirs(model_dir, exist_ok=True)

    # ---------- 1. Load or prepare data ----------
    train_path = os.path.join(DATA_PROCESSED, 'train.csv')
    if not os.path.exists(train_path):
        print("No preprocessed data found. Loading raw dataset...")
        df = load_raw_dataset()
        df = remove_duplicates(df, subset='text')
        train_df, val_df, test_df = split_dataset(df)
    else:
        print("Loading preprocessed splits...")
        train_df = load_split('train')
        val_df = load_split('val')
        test_df = load_split('test')

    # ---------- 2. Preprocess ----------
    print("\nPreprocessing training data (baseline cleaning)...")
    train_df = preprocess_dataset(train_df, method='baseline')
    val_df = preprocess_dataset(val_df, method='baseline')
    test_df = preprocess_dataset(test_df, method='baseline')

    X_train = train_df['clean_text'].values
    y_train = train_df['label'].values
    X_val = val_df['clean_text'].values
    y_val = val_df['label'].values
    X_test = test_df['clean_text'].values
    y_test = test_df['label'].values

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---------- 3. TF-IDF Vectorization ----------
    print("\nBuilding TF-IDF features...")
    vectorizer, X_train_tfidf = build_tfidf_vectorizer(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    save_vectorizer(vectorizer, model_dir)

    # ---------- 4. Train Naive Bayes ----------
    print("\nTraining Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='f1')
    print(f"5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ---------- 5. Save model ----------
    model_path = os.path.join(model_dir, 'naive_bayes.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    # ---------- 6. Evaluate ----------
    print("\n" + "=" * 60)
    print("VALIDATION SET EVALUATION")
    print("=" * 60)
    y_val_pred = model.predict(X_val_tfidf)
    y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]
    full_evaluation(y_val, y_val_pred, y_val_proba, model_dir,
                    prefix='val', model_name='Baseline (TF-IDF + NB)')

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    y_test_pred = model.predict(X_test_tfidf)
    y_test_proba = model.predict_proba(X_test_tfidf)[:, 1]
    full_evaluation(y_test, y_test_pred, y_test_proba, model_dir,
                    prefix='test', model_name='Baseline (TF-IDF + NB)')

    # ---------- 7. Feature importance ----------
    feature_names = vectorizer.get_feature_names_out()
    log_probs = model.feature_log_prob_

    # Top words for Fake (class 0) and Real (class 1)
    top_n = 15
    print(f"\nTop {top_n} words for FAKE news:")
    fake_top = np.argsort(log_probs[0])[-top_n:][::-1]
    for idx in fake_top:
        print(f"  {feature_names[idx]:25s} (log-prob: {log_probs[0][idx]:.4f})")

    print(f"\nTop {top_n} words for REAL news:")
    real_top = np.argsort(log_probs[1])[-top_n:][::-1]
    for idx in real_top:
        print(f"  {feature_names[idx]:25s} (log-prob: {log_probs[1][idx]:.4f})")

    print("\nBaseline training complete!")
    return model, vectorizer


if __name__ == '__main__':
    train_baseline()
