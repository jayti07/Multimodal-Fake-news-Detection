"""
Text Fake News Classifier Module
Supports baseline TF-IDF + Logistic Regression and fine-tuned Transformer (DistilBERT).
"""

import os
import joblib
from typing import Dict, Any

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))


class TextFakeNewsClassifier:
    def __init__(self, model_name_or_path: str = "distilbert-base-uncased"):
        self.model_name_or_path = model_name_or_path
        self.vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
        self.clf_path = os.path.join(MODEL_DIR, "text_baseline_logistic_regression.joblib")

        self.vectorizer = None
        self.clf = None

        if os.path.exists(self.vectorizer_path) and os.path.exists(self.clf_path):
            try:
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.clf = joblib.load(self.clf_path)
            except Exception as e:
                print(f"[TextClassifier Warning] Failed to load trained model: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies input text claim as Real or Fake with confidence score.
        """
        if not text or not text.strip():
            return {"label": "Unknown", "confidence": 0.0, "error": "Empty text provided"}

        if self.vectorizer is not None and self.clf is not None:
            vec = self.vectorizer.transform([text])
            prob_fake = float(self.clf.predict_proba(vec)[0][1])
            label = "Fake" if prob_fake > 0.5 else "Real"
            confidence = round(max(prob_fake, 1.0 - prob_fake) * 100, 2)

            return {
                "label": label,
                "confidence": confidence,
                "fake_probability": round(prob_fake, 4),
                "model_used": "AI Text Style & Pattern Classifier"
            }


        # Fallback heuristic
        return {
            "label": "Unverified",
            "confidence": 50.0,
            "fake_probability": 0.50,
            "model_used": "Baseline Fallback"
        }
