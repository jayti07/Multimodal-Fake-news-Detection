"""
Academic Research Paper Evaluation & Visualization Script
Generates Classification Report, Confusion Matrix PNG, and ROC-AUC Curve PNG for Text Fake News Classifier.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
import joblib

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_assets"))


def generate_research_figures():
    os.makedirs(PAPER_DIR, exist_ok=True)
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print(f"Error: Datasets not found in {DATA_DIR}")
        return

    print("Loading ISOT dataset for academic paper evaluation...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["label"] = 1  # Fake
    df_true["label"] = 0  # Real

    df_fake["content"] = df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
    df_true["content"] = df_true["title"].fillna("") + " " + df_true["text"].fillna("")

    df = pd.concat([df_fake[["content", "label"]], df_true[["content", "label"]]], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Extracting TF-IDF features and predicting...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print("\n" + "=" * 60)
    print("      RESEARCH PAPER ACADEMIC EVALUATION METRICS TABLE      ")
    print("=" * 60)
    print(f" Total Dataset Samples : {len(df):,}")
    print(f" Training Samples (80%): {len(X_train):,}")
    print(f" Testing Samples  (20%): {len(X_test):,}")
    print("-" * 60)
    print(f" Overall Accuracy      : {acc * 100:.4f}%")
    print(f" Precision (Fake News) : {prec * 100:.4f}%")
    print(f" Recall (Fake News)    : {rec * 100:.4f}%")
    print(f" F1-Score (Fake News)  : {f1 * 100:.4f}%")
    print("-" * 60)
    print("\nFULL CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["Real News (0)", "Fake News (1)"], digits=4))
    print("=" * 60)

    # 1. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], annot_kws={"size": 14})
    plt.title("Confusion Matrix - Fake News Text Classifier", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("True Label", fontsize=12, labelpad=8)
    plt.tight_layout()
    cm_path = os.path.join(PAPER_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix Plot to: {cm_path}")

    # 2. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    roc_path = os.path.join(PAPER_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve Plot to: {roc_path}")


if __name__ == "__main__":
    generate_research_figures()
