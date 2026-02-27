"""
BERT (DistilBERT) fine-tuning for Fake News Detection.

This script can run locally (slow on CPU) or on Google Colab (recommended).
Run: python -m src.train_bert

For Colab: Copy this script into a notebook cell and run with GPU runtime.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_split, MODELS_DIR, DATA_PROCESSED, load_raw_dataset, split_dataset
from src.preprocessing import preprocess_dataset, remove_duplicates
from src.evaluate import full_evaluation


def train_bert(model_name='distilbert-base-uncased', max_length=512,
               epochs=3, batch_size=8, learning_rate=2e-5):
    """
    Fine-tune DistilBERT for fake news classification.

    For Google Colab:
        1. Upload data/processed/train.csv, val.csv, test.csv to Drive
        2. Set batch_size=16 (GPU can handle more)
        3. Run this function
    """
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

    model_dir = os.path.join(MODELS_DIR, 'bert')
    os.makedirs(model_dir, exist_ok=True)

    # ---------- 1. Load data ----------
    train_path = os.path.join(DATA_PROCESSED, 'train.csv')
    if not os.path.exists(train_path):
        print("No preprocessed data found. Loading raw dataset...")
        df = load_raw_dataset()
        df = remove_duplicates(df, subset='text')
        split_dataset(df)

    print("Loading data splits...")
    train_df = load_split('train')
    val_df = load_split('val')
    test_df = load_split('test')

    # ---------- 2. Preprocess (minimal for BERT) ----------
    print("\nPreprocessing for BERT (minimal)...")
    train_df = preprocess_dataset(train_df, method='bert')
    val_df = preprocess_dataset(val_df, method='bert')
    test_df = preprocess_dataset(test_df, method='bert')

    # ---------- 3. Tokenize ----------
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_texts(texts, max_len):
        return tokenizer(
            texts.tolist(),
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )

    print("Tokenizing datasets...")
    train_encodings = tokenize_texts(train_df['clean_text'].values, max_length)
    val_encodings = tokenize_texts(val_df['clean_text'].values, max_length)
    test_encodings = tokenize_texts(test_df['clean_text'].values, max_length)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # ---------- 4. Create TF Datasets ----------
    def create_dataset(encodings, labels, batch_sz, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels
        ))
        if shuffle:
            dataset = dataset.shuffle(10000)
        return dataset.batch(batch_sz).prefetch(tf.data.AUTOTUNE)

    train_dataset = create_dataset(train_encodings, y_train, batch_size, shuffle=True)
    val_dataset = create_dataset(val_encodings, y_val, batch_size)
    test_dataset = create_dataset(test_encodings, y_test, batch_size)

    # ---------- 5. Load Model ----------
    print(f"\nLoading model: {model_name}")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # ---------- 6. Compile ----------
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # ---------- 7. Train ----------
    print(f"\nTraining for {epochs} epochs (batch_size={batch_size})...")
    print("NOTE: This is VERY slow on CPU. Use Google Colab GPU for faster training.")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)

    # ---------- 8. Save Model ----------
    print(f"\nSaving model to {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # ---------- 9. Evaluate ----------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Get predictions
    predictions = model.predict(test_dataset)
    logits = predictions.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    y_test_pred = np.argmax(probs, axis=1)
    y_test_proba = probs[:, 1]  # Probability of Real class

    full_evaluation(
        y_test, y_test_pred, y_test_proba,
        model_dir, prefix='test', model_name='DistilBERT'
    )

    print("\nBERT fine-tuning complete!")
    return model, tokenizer


if __name__ == '__main__':
    train_bert()
