# 🛡️ Multi-Modal Fake News & Deepfake Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-fake-news-detection-0102.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

An end-to-end Multi-Modal AI system to detect fake news across **Text**, **Video (Deepfake)**, and **Audio (Synthetic Voice)** modalities, featuring an automated Retrieval-Augmented Generation (RAG) live fact-verification engine.

🌐 **Live Interactive Web Application**: [https://multimodal-fake-news-detection-0102.streamlit.app](https://multimodal-fake-news-detection-0102.streamlit.app)

---


## 🌟 Key Features

- 📰 **Text News Claim & Fact Verification**: TF-IDF + Logistic Regression classifier trained on **44,898 ISOT news articles** (**99.00% Accuracy**) coupled with a live RAG pipeline (Google News RSS + Domain Whitelisting for 26+ trusted sources & Snopes refutation stance detection).
- 🎥 **Video Deepfake Detection**: Computer Vision pipeline using OpenCV + MTCNN (facenet-pytorch) neural network face cropping with 20% spatial margin + PyTorch **EfficientNet-B0** classifier trained on Celeb-DF v2 benchmark (**98.75% Accuracy**).
- 🎙️ **Audio Voice Authenticity**: Librosa Mel-Spectrogram 2D matrix extraction + PyTorch 2D Spectrogram CNN trained on ASVspoof 2019 dataset (**99.90% Accuracy**).
- 🗣️ **Automatic Speech Recognition (ASR)**: Integrated **OpenAI Whisper** ASR (100% offline PyTorch Speech-to-Text model) for automatic audio transcription from video/audio clips.
- 🎨 **Interactive Streamlit Web Dashboard**: Vibrant, modern web UI for single and multi-modal analysis.

---

## 📊 Performance Summary

| Modality | Model Architecture | Benchmark Dataset | Overall Test Accuracy | ROC-AUC ($\mathbf{AUC}$) |
| :--- | :--- | :--- | :--- | :--- |
| 📰 **Text Claim** | TF-IDF + Logistic Regression | ISOT Fake News Dataset (44,898 articles) | **99.00%** | **0.9992** |
| 🎙️ **Audio Voice** | PyTorch 2D Mel-Spectrogram CNN | ASVspoof 2019 (LA Track) | **99.90%** | **0.9998** |
| 🎥 **Video Deepfake** | MTCNN + PyTorch EfficientNet-B0 | Celeb-DF v2 Benchmark Dataset | **98.75%** | **0.9972** |

---

## 📁 Repository Structure

```
multimodal-fake-news-detector/
├── app/
│   └── app.py                        # Streamlit Interactive Web Application
├── models/                           # Saved Model Weights (.pth, .joblib)
│   ├── text_baseline_logistic_regression.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── audio_spoof_cnn.pth
│   └── video_deepfake_efficientnet.pth
├── paper_assets/                     # High-Resolution 300 DPI Research Figures
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── audio_confusion_matrix.png
│   ├── audio_roc_curve.png
│   ├── video_confusion_matrix.png
│   └── video_roc_curve.png
├── src/
│   ├── modalities/
│   │   ├── text_classifier.py        # TF-IDF + Logistic Regression Classifier
│   │   ├── fact_checker.py           # Live Domain-Whitelisted RAG Engine
│   │   ├── audio_classifier.py       # Audio CNN + OpenAI Whisper ASR
│   │   └── video_classifier.py       # MTCNN + EfficientNet Deepfake Classifier
│   ├── train_text_model.py           # Text Training Script
│   ├── train_audio_model.py          # Audio CNN Training Script
│   ├── train_video_model.py          # Video EfficientNet Training Script
│   ├── generate_metrics.py           # Text Research Figures Generator
│   ├── generate_audio_paper_metrics.py# Audio Research Figures Generator
│   └── generate_video_paper_metrics.py# Video Research Figures Generator
├── requirements.txt
└── README.md
```

