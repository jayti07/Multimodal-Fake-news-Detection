"""
Training Script for Text Fake News Classifier
Trains baseline TF-IDF + Logistic Regression and fine-tunes DistilBERT.
Dataset location: data/text/Fake.csv and data/text/True.csv
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove publisher headers like "WASHINGTON (Reuters) - " or "(Reuters)"
    text = re.sub(r"^.*?\([^)]*Reuters[^)]*\)\s*[-–—]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bReuters\b", "", text, flags=re.IGNORECASE)
    # Basic lowercasing & whitespace normalization
    text = text.lower().strip()
    return text


def load_and_preprocess_data():
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            f"Please download Fake.csv and True.csv from Kaggle and place them in: {DATA_DIR}"
        )

    print("Loading datasets...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Clean text to remove spurious publisher signatures (e.g. 'Reuters')
    print("Cleaning publisher artifacts (removing 'Reuters' headers)...")
    df_fake["clean_text"] = df_fake["text"].apply(clean_text)
    df_true["clean_text"] = df_true["text"].apply(clean_text)

    # Assign binary labels: 1 for Fake, 0 for Real
    df_fake["label"] = 1
    df_true["label"] = 0

    # Combine title and cleaned text
    df_fake["content"] = df_fake["title"].fillna("") + " " + df_fake["clean_text"]
    df_true["content"] = df_true["title"].fillna("") + " " + df_true["clean_text"]

    # Merge into a single dataframe
    df = pd.concat([df_fake[["content", "label"]], df_true[["content", "label"]]], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Total cleaned samples: {len(df)} (Fake: {len(df_fake)}, Real: {len(df_true)})")
    return df



def train_baseline_model(df):
    print("\n--- Training Baseline Model (TF-IDF + Logistic Regression) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Baseline Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "text_baseline_logistic_regression.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    print(f"Saved baseline model and vectorizer to: {MODEL_DIR}")


if __name__ == "__main__":
    df = load_and_preprocess_data()
    train_baseline_model(df)
