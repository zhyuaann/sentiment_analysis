# Sentiment Analysis

This project is a **sentiment analysis** classification system of text comments (in English) using **TF-IDF vectorization** and **Logistic Regression**. The model is trained to classify comments into three sentiment categories: **Negative (0), Neutral (1), and Positive (2)**.

---

## Feature

- Automatic text preprocessing (lowercase, remove URLs, mentions, numbers, symbols, etc.)
- Feature extraction using TF-IDF (with n-grams and frequency filters)
- Hyperparameter tuning with `RandomizedSearchCV`
- Model evaluation: accuracy, F1-score, precision, recall, MCC, confusion matrix
- Simple Flask-based UI for comment input

## Model and Evaluation

Recent Best Model: **Logistic Regression + TF-IDF (8000 feature, ngram (1,2))**

