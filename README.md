# Multi-Modal Fake News & Deepfake Detection System

A complete AI/ML system to classify fake news across **Text**, **Video (Deepfake)**, and **Audio (Synthetic Voice)**, featuring a Retrieval-Augmented Generation (RAG) fact-verification engine.

---

## 🌟 Key Features
- **Text Detection & Fact Verification**: Fine-tuned DistilBERT classifier paired with a RAG pipeline (Sentence-BERT + Google Fact Check / News API) for real article attribution.
- **Video Deepfake Detection**: MTCNN face cropping + EfficientNet-B0 frame feature extraction with temporal aggregation.
- **Audio Synthetic Voice Detection**: Librosa Mel-Spectrogram extraction + 2D CNN (ResNet-18) voice spoof classifier.
- **Interactive UI & REST API**: Streamlit dashboard for single/multimodal analysis and FastAPI backend endpoints.

---

## 📁 Repository Structure
```
multimodal-fake-news-detector/
├── data/                      # Dataset helpers and test media samples
├── src/                       # Core ML modules
│   ├── modalities/
│   │   ├── text_classifier.py # Text model (DistilBERT / TF-IDF)
│   │   ├── fact_checker.py    # RAG Fact Verification Engine
│   │   ├── video_classifier.py# OpenCV/MTCNN + EfficientNet Video Deepfake
│   │   └── audio_classifier.py# Librosa + ResNet-18 Audio Spoof Classifier
│   └── fusion/
│       └── late_fusion.py     # Multi-modal score aggregation
├── models/                    # Saved weights (.pth, .onnx, tokenizers)
├── app/                       # Application layer
│   ├── app.py                 # Streamlit Web App
│   └── api.py                 # FastAPI Backend API
├── .env.example
├── requirements.txt
└── README.md
```
