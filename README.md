# Fake News Detection using NLP

A machine learning project that detects fake news articles using Natural Language Processing techniques. Built with TF-IDF + Naive Bayes (baseline), LSTM, and BERT models.

## Features

- **Multiple ML Models**: TF-IDF + Naive Bayes (95.7% accuracy), LSTM, DistilBERT
- **Streamlit Web App**: Interactive UI to analyze news articles
- **Explainability**: See which words contribute to the prediction
- **Sentiment Analysis**: VADER-based sentiment scoring
- **Bias Detection**: Political leaning detection (Left/Center/Right)
- **Source Credibility**: Domain reputation scoring
- **Admin Panel**: View prediction logs, manage models

## Project Structure

```
fake-news-detection/
├── data/
│   ├── raw/                    # Place Fake.csv + True.csv here
│   └── processed/              # Auto-generated train/val/test splits
├── src/
│   ├── preprocessing.py        # Text cleaning (3 levels)
│   ├── feature_engineering.py  # TF-IDF, tokenizers
│   ├── train_baseline.py       # TF-IDF + Naive Bayes training
│   ├── train_lstm.py           # LSTM with attention
│   ├── train_bert.py           # DistilBERT fine-tuning
│   ├── predict.py              # Unified prediction interface
│   ├── evaluate.py             # Metrics and visualizations
│   ├── sentiment.py            # VADER sentiment analysis
│   └── explainability.py       # Model explanations
├── models/                     # Trained model artifacts
├── app/
│   └── app.py                  # Streamlit application
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fathimavafiya-netizen/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## Dataset

Download the **ISOT Fake News Dataset** from Kaggle:
- [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `Fake.csv` and `True.csv` in `data/raw/`.

## Usage

### Train the baseline model:
```bash
python -m src.train_baseline
```

### Train LSTM model:
```bash
python -m src.train_lstm
```

### Train BERT (recommended on Google Colab with GPU):
```bash
python -m src.train_bert
```

### Launch the web app:
```bash
streamlit run app/app.py
```

Then open http://localhost:8501 in your browser.

## Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| TF-IDF + Naive Bayes | 95.69% | 0.9569 | 0.9899 |
| LSTM + Attention | ~93% | ~0.93 | ~0.97 |
| DistilBERT | ~96% | ~0.96 | ~0.98 |

## Web App Pages

1. **Prediction**: Paste text or upload .txt file to analyze
2. **Model Performance**: View metrics, confusion matrix, ROC curves
3. **Explainability Dashboard**: Word importance visualization
4. **Admin Panel**: Prediction logs, model management (password: `admin123`)

## Tech Stack

- **ML/NLP**: scikit-learn, NLTK, TensorFlow, Transformers
- **Web**: Streamlit
- **Visualization**: Matplotlib, Seaborn, Plotly

## License

MIT License
