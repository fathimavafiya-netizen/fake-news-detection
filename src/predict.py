"""
Unified prediction interface with ModelRegistry.
Supports Baseline (NB), LSTM, and BERT models.
"""

import os
import pickle
import json
import numpy as np

from src.preprocessing import clean_text_baseline, clean_text_lstm, clean_text_bert
from src.utils import MODELS_DIR


class BaselinePredictor:
    """TF-IDF + Naive Bayes predictor."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.join(MODELS_DIR, 'baseline')
        self.model = None
        self.vectorizer = None
        self.load()

    def load(self):
        model_path = os.path.join(self.model_dir, 'naive_bayes.pkl')
        vec_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Baseline model not found at {model_path}. Train it first.")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text):
        """
        Predict whether text is fake or real.

        Returns:
            dict with 'label', 'confidence', 'probabilities'
        """
        clean = clean_text_baseline(text)
        if not clean:
            return {'label': 'Unknown', 'confidence': 0.0,
                    'probabilities': {'Fake': 0.5, 'Real': 0.5}}

        X = self.vectorizer.transform([clean])
        proba = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))

        label = 'Real' if pred_class == 1 else 'Fake'
        confidence = float(proba[pred_class])

        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {'Fake': float(proba[0]), 'Real': float(proba[1])}
        }

    def get_feature_weights(self, text, top_n=10):
        """Get top TF-IDF features contributing to prediction."""
        clean = clean_text_baseline(text)
        if not clean:
            return []

        X = self.vectorizer.transform([clean])
        feature_names = self.vectorizer.get_feature_names_out()

        # Get non-zero features
        nonzero = X.nonzero()[1]
        tfidf_scores = X.toarray()[0]

        # Multiply by NB log probabilities to get contribution
        log_probs = self.model.feature_log_prob_
        diff = log_probs[1] - log_probs[0]  # Real - Fake

        contributions = []
        for idx in nonzero:
            word = feature_names[idx]
            score = float(tfidf_scores[idx] * diff[idx])
            contributions.append((word, score))

        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:top_n]


class LSTMPredictor:
    """LSTM model predictor."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.join(MODELS_DIR, 'lstm')
        self.model = None
        self.word_index = None
        self.config = None
        self.max_length = 500

    def load(self):
        import tensorflow as tf

        model_path = os.path.join(self.model_dir, 'lstm_model.keras')
        if not os.path.exists(model_path):
            # Try .h5 format
            model_path = os.path.join(self.model_dir, 'lstm_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found in {self.model_dir}. Train it first.")

        self.model = tf.keras.models.load_model(model_path)

        with open(os.path.join(self.model_dir, 'word_index.json'), 'r') as f:
            self.word_index = json.load(f)
        with open(os.path.join(self.model_dir, 'tokenizer_config.json'), 'r') as f:
            self.config = json.load(f)
            self.max_length = self.config.get('max_length', 500)

    def _text_to_sequence(self, text):
        oov_idx = self.word_index.get('<OOV>', 1)
        seq = [self.word_index.get(w, oov_idx) for w in text.split()]
        # Pad/truncate
        padded = np.zeros((1, self.max_length), dtype=np.int32)
        length = min(len(seq), self.max_length)
        padded[0, :length] = seq[:length]
        return padded

    def predict(self, text):
        if self.model is None:
            self.load()

        clean = clean_text_lstm(text)
        if not clean:
            return {'label': 'Unknown', 'confidence': 0.0,
                    'probabilities': {'Fake': 0.5, 'Real': 0.5}}

        X = self._text_to_sequence(clean)
        proba_real = float(self.model.predict(X, verbose=0)[0][0])
        proba_fake = 1.0 - proba_real

        label = 'Real' if proba_real >= 0.5 else 'Fake'
        confidence = proba_real if label == 'Real' else proba_fake

        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {'Fake': proba_fake, 'Real': proba_real}
        }


class BERTPredictor:
    """DistilBERT model predictor."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.join(MODELS_DIR, 'bert')
        self.model = None
        self.tokenizer = None
        self.max_length = 512

    def load(self):
        from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

        if not os.path.exists(self.model_dir) or not os.listdir(self.model_dir):
            raise FileNotFoundError(
                f"BERT model not found in {self.model_dir}. "
                "Train it on Google Colab first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_dir)

    def predict(self, text):
        if self.model is None:
            self.load()

        clean = clean_text_bert(text)
        if not clean:
            return {'label': 'Unknown', 'confidence': 0.0,
                    'probabilities': {'Fake': 0.5, 'Real': 0.5}}

        inputs = self.tokenizer(
            clean, return_tensors='tf',
            max_length=self.max_length, truncation=True, padding=True
        )
        outputs = self.model(inputs)
        import tensorflow as tf
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

        pred_class = int(np.argmax(probs))
        label = 'Real' if pred_class == 1 else 'Fake'
        confidence = float(probs[pred_class])

        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {'Fake': float(probs[0]), 'Real': float(probs[1])}
        }

    def get_attention_weights(self, text):
        """Extract attention weights for explainability."""
        if self.model is None:
            self.load()

        clean = clean_text_bert(text)
        inputs = self.tokenizer(
            clean, return_tensors='tf',
            max_length=self.max_length, truncation=True, padding=True
        )
        outputs = self.model(inputs, output_attentions=True)

        # Average attention across all heads of last layer
        last_layer_attn = outputs.attentions[-1].numpy()[0]  # (heads, seq, seq)
        avg_attn = last_layer_attn.mean(axis=0)  # (seq, seq)

        # CLS token attention to all other tokens
        cls_attn = avg_attn[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].numpy())

        return list(zip(tokens, cls_attn.tolist()))


class ModelRegistry:
    """Factory to manage and load available models."""

    _predictors = {}

    @classmethod
    def get_available_models(cls):
        """List models that have trained artifacts."""
        available = []
        baseline_path = os.path.join(MODELS_DIR, 'baseline', 'naive_bayes.pkl')
        if os.path.exists(baseline_path):
            available.append('baseline')

        lstm_dir = os.path.join(MODELS_DIR, 'lstm')
        if os.path.exists(os.path.join(lstm_dir, 'lstm_model.keras')) or \
           os.path.exists(os.path.join(lstm_dir, 'lstm_model.h5')):
            available.append('lstm')

        bert_dir = os.path.join(MODELS_DIR, 'bert')
        if os.path.exists(bert_dir) and any(
            f.endswith(('.bin', '.h5', '.index', 'config.json'))
            for f in os.listdir(bert_dir) if os.path.isfile(os.path.join(bert_dir, f))
        ):
            available.append('bert')

        return available

    @classmethod
    def get_predictor(cls, model_type='baseline'):
        """Get or create a predictor instance (cached)."""
        if model_type not in cls._predictors:
            if model_type == 'baseline':
                cls._predictors[model_type] = BaselinePredictor()
            elif model_type == 'lstm':
                predictor = LSTMPredictor()
                predictor.load()
                cls._predictors[model_type] = predictor
            elif model_type == 'bert':
                predictor = BERTPredictor()
                predictor.load()
                cls._predictors[model_type] = predictor
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return cls._predictors[model_type]

    @classmethod
    def predict(cls, text, model_type='baseline'):
        """Make a prediction using the specified model."""
        predictor = cls.get_predictor(model_type)
        return predictor.predict(text)
