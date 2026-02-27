"""
Utility functions: data loading, splitting, logging, and helpers.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')


def load_raw_dataset():
    """
    Load the ISOT Fake News Dataset from data/raw/.
    Supports two formats:
      1. Separate files: Fake.csv + True.csv
      2. Single file with 'label' column
    Returns:
        DataFrame with columns: title, text, label (0=Fake, 1=Real)
    """
    fake_path = os.path.join(DATA_RAW, 'Fake.csv')
    true_path = os.path.join(DATA_RAW, 'True.csv')

    if os.path.exists(fake_path) and os.path.exists(true_path):
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 0
        true_df = pd.read_csv(true_path)
        true_df['label'] = 1

        df = pd.concat([fake_df, true_df], ignore_index=True)
        print(f"Loaded ISOT dataset: {len(fake_df)} fake + {len(true_df)} real = {len(df)} total")
    else:
        # Try single CSV with label column
        csv_files = [f for f in os.listdir(DATA_RAW) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {DATA_RAW}. "
                "Please download the ISOT Fake News Dataset from Kaggle "
                "and place Fake.csv and True.csv in data/raw/"
            )
        df = pd.read_csv(os.path.join(DATA_RAW, csv_files[0]))
        if 'label' not in df.columns:
            raise ValueError("Dataset must have a 'label' column (0=Fake, 1=Real)")
        print(f"Loaded dataset from {csv_files[0]}: {len(df)} articles")

    # Ensure required columns exist
    if 'title' not in df.columns:
        df['title'] = ''
    if 'text' not in df.columns:
        raise ValueError("Dataset must have a 'text' column")

    # Keep only needed columns
    df = df[['title', 'text', 'label']].copy()
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')

    return df


def split_dataset(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Stratified train/val/test split.
    Saves splits to data/processed/.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size),
        stratify=df['label'], random_state=random_state
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio),
        stratify=temp_df['label'], random_state=random_state
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_PROCESSED, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_PROCESSED, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_PROCESSED, 'test.csv'), index=False)

    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts(normalize=True)}")
    print(f"Saved to {DATA_PROCESSED}")

    return train_df, val_df, test_df


def load_split(split_name='train'):
    """Load a saved split (train/val/test) from data/processed/."""
    path = os.path.join(DATA_PROCESSED, f'{split_name}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run data splitting first.")
    return pd.read_csv(path)


def save_metrics(metrics_dict, model_dir):
    """Save evaluation metrics to JSON file."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"Metrics saved to {path}")


def log_prediction(text_preview, prediction, confidence, model_type, sentiment=None):
    """Append prediction to logs/predictions.csv."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, 'predictions.csv')

    entry = {
        'timestamp': datetime.now().isoformat(),
        'text_preview': text_preview[:200],
        'prediction': prediction,
        'confidence': round(confidence, 4),
        'model_type': model_type,
        'sentiment_compound': sentiment if sentiment else None
    }

    log_df = pd.DataFrame([entry])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
