"""
LSTM model training with attention mechanism.
Run: python -m src.train_lstm
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_split, MODELS_DIR, DATA_PROCESSED, load_raw_dataset, split_dataset
from src.preprocessing import preprocess_dataset, remove_duplicates
from src.feature_engineering import (
    build_lstm_tokenizer, texts_to_sequences,
    save_lstm_tokenizer, load_lstm_tokenizer
)
from src.evaluate import full_evaluation


# ---- Custom Attention Layer ----
def build_attention_layer():
    """Returns a custom Attention layer class for Keras."""
    import tensorflow as tf

    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name='attention_weight',
                shape=(input_shape[-1], 1),
                initializer='glorot_uniform',
                trainable=True
            )
            self.b = self.add_weight(
                name='attention_bias',
                shape=(input_shape[1], 1),
                initializer='zeros',
                trainable=True
            )

        def call(self, x):
            import tensorflow as tf
            e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
            a = tf.nn.softmax(e, axis=1)
            output = tf.reduce_sum(x * a, axis=1)
            return output

        def get_config(self):
            return super().get_config()

    return AttentionLayer


def build_lstm_model(vocab_size, embedding_dim=128, lstm_units=128,
                     max_length=500, dropout_rate=0.3):
    """
    Build Bidirectional LSTM with Attention.

    Architecture:
        Embedding -> BiLSTM -> Attention -> Dense -> Dropout -> Output
    """
    import tensorflow as tf

    AttentionLayer = build_attention_layer()

    inputs = tf.keras.Input(shape=(max_length,), dtype='int32')
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        mask_zero=False
    )(inputs)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
    )(x)
    x = AttentionLayer()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def train_lstm(max_length=500, vocab_size=20000, embedding_dim=128,
               lstm_units=128, epochs=15, batch_size=32):
    """Train the LSTM model end-to-end."""

    model_dir = os.path.join(MODELS_DIR, 'lstm')
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

    # ---------- 2. Preprocess ----------
    print("\nPreprocessing for LSTM...")
    train_df = preprocess_dataset(train_df, method='lstm')
    val_df = preprocess_dataset(val_df, method='lstm')
    test_df = preprocess_dataset(test_df, method='lstm')

    X_train_text = train_df['clean_text'].values
    y_train = train_df['label'].values.astype(np.float32)
    X_val_text = val_df['clean_text'].values
    y_val = val_df['label'].values.astype(np.float32)
    X_test_text = test_df['clean_text'].values
    y_test = test_df['label'].values.astype(np.float32)

    print(f"Train: {len(X_train_text)}, Val: {len(X_val_text)}, Test: {len(X_test_text)}")

    # ---------- 3. Build tokenizer ----------
    print("\nBuilding tokenizer...")
    word_index, config = build_lstm_tokenizer(X_train_text, vocab_size=vocab_size)
    config['max_length'] = max_length
    save_lstm_tokenizer(word_index, config, model_dir)

    actual_vocab_size = config['vocab_size']

    # ---------- 4. Convert to sequences ----------
    print("Converting texts to sequences...")
    X_train = texts_to_sequences(X_train_text, word_index, max_length)
    X_val = texts_to_sequences(X_val_text, word_index, max_length)
    X_test = texts_to_sequences(X_test_text, word_index, max_length)

    print(f"Sequence shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---------- 5. Build model ----------
    print("\nBuilding LSTM model...")
    import tensorflow as tf

    model = build_lstm_model(
        vocab_size=actual_vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        max_length=max_length
    )
    model.summary()

    # ---------- 6. Callbacks ----------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'lstm_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
    ]

    # ---------- 7. Train ----------
    print(f"\nTraining for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)

    # ---------- 8. Evaluate ----------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    y_test_proba = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    full_evaluation(
        y_test.astype(int), y_test_pred, y_test_proba,
        model_dir, prefix='test', model_name='LSTM + Attention'
    )

    print("\nLSTM training complete!")
    return model


if __name__ == '__main__':
    train_lstm()
