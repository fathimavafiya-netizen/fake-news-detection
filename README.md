 Fake News Detection System

A complete end-to-end **Machine Learning + NLP project** that detects whether a news article is **Fake or Real** using text classification techniques.

---

 Project Overview

This project builds a **Fake News Detection System** using:

* Natural Language Processing (NLP)
* Machine Learning (TF-IDF + Naive Bayes)
* Interactive Web App using Streamlit

The system takes a news article as input and predicts whether it is **Fake  or Real **.

---

Features

 Text preprocessing and cleaning
 TF-IDF vectorization
 Multinomial Naive Bayes model
 Model evaluation (Accuracy, Precision, Recall, F1-score)
 Interactive Streamlit web interface
 Explainability dashboard (basic insights)

---

Model Performance

* **Accuracy:** ~95.6%
* **F1 Score:** ~95.6%
* **ROC-AUC:** ~0.99

Strong performance on both validation and test datasets.

---

Project Structure

```
fake-news-detection/
│
├── app/                  # Streamlit web app
│   └── app.py
│
├── src/                  # Core ML pipeline
│   └── train_baseline.py
│
├── models/               # Saved models (ignored in Git)
│
├── data/
│   ├── raw/              # Raw dataset (not included)
│   └── processed/        # Processed data
│
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── .gitignore
```

---

Dataset

This project uses the **Fake and Real News Dataset**:

* Fake news articles
* Real news articles

Download from:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

Installation & Setup

Clone the Repository

```
git clone https://github.com/fathimavafiya-netizen/fake-news-detection.git
cd fake-news-detection
```

---

Install Dependencies

```
pip install -r requirements.txt
```

---

Add Dataset

Create the following folder:

```
data/raw/
```

Place files inside:

```
Fake.csv
True.csv
```

---

Train the Model

```
python -m src.train_baseline
```

---

Run the Application

```
streamlit run app/app.py
```

Then open:

```
http://localhost:8501
```

---
Application Preview

* Enter a news headline or article
* Click **Predict**
* Get output: **Fake or Real**

---

Technologies Used

* Python 
* Pandas & NumPy
* Scikit-learn
* NLTK
* Streamlit
* Matplotlib & Seaborn

---

Future Improvements

* 🔹 Deep Learning models (LSTM / BERT)
* 🔹 Real-time news API integration
* 🔹 Better UI/UX design
* 🔹 Deployment on cloud (AWS / Streamlit Cloud)

---


Acknowledgements

* Kaggle dataset contributors
* Open-source ML community

---

Conclusion

This project demonstrates a **complete ML workflow**:

Data Collection
Preprocessing
Model Training
Evaluation
Deployment

