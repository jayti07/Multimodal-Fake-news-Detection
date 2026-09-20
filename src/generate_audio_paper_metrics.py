"""
Academic Research Paper Evaluation & Visualization Script for Model 2 (Audio Synthetic Voice Detector)
Generates Classification Metrics Table, Confusion Matrix PNG, and ROC-AUC Curve PNG.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)

from train_audio_model import extract_mel_spectrogram, SimpleAudioCNN, DATA_DIR, MODEL_DIR

PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_assets"))


class BalancedASVspoofLADataset(Dataset):
    """
    Balanced Dataset Loader for ASVspoof 2019 LA directory.
    Loads equal counts of bonafide (0) and spoof (1) samples.
    """
    def __init__(self, la_dir_path, max_per_class=500):
        self.samples = []
        
        protocol_file = None
        for root, dirs, files in os.walk(la_dir_path):
            for f in files:
                if f.endswith(".txt") and ".cm." in f:
                    protocol_file = os.path.join(root, f)
                    break
            if protocol_file:
                break

        audio_dir = None
        for root, dirs, files in os.walk(la_dir_path):
            if any(f.endswith(('.flac', '.wav')) for f in files):
                audio_dir = root
                break

        if protocol_file and audio_dir:
            with open(protocol_file, 'r') as f:
                lines = f.readlines()

            real_count = 0
            fake_count = 0

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    file_id = parts[1]
                    key = parts[-1].lower()
                    
                    if key not in ["bonafide", "spoof"]:
                        continue

                    label = 0 if key == "bonafide" else 1

                    if label == 0 and real_count >= max_per_class:
                        continue
                    if label == 1 and fake_count >= max_per_class:
                        continue

                    flac_path = os.path.join(audio_dir, f"{file_id}.flac")
                    wav_path = os.path.join(audio_dir, f"{file_id}.wav")

                    if os.path.exists(flac_path):
                        self.samples.append((flac_path, label))
                        if label == 0: real_count += 1
                        else: fake_count += 1
                    elif os.path.exists(wav_path):
                        self.samples.append((wav_path, label))
                        if label == 0: real_count += 1
                        else: fake_count += 1

                    if real_count >= max_per_class and fake_count >= max_per_class:
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        tensor_spec = extract_mel_spectrogram(audio_path)
        if tensor_spec is None:
            tensor_spec = torch.zeros((1, 128, 128), dtype=torch.float32)
        return tensor_spec, torch.tensor(label, dtype=torch.long)


def generate_audio_research_figures():
    os.makedirs(PAPER_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = os.path.join(MODEL_DIR, "audio_spoof_cnn.pth")
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}")
        return

    # 1. Load trained PyTorch CNN Model
    model = SimpleAudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Balanced Evaluation Dataset
    print("Loading ASVspoof 2019 balanced benchmark dataset for academic evaluation...")
    la_path = r"C:\Users\jayti\Downloads\3rdModel\LA\LA"
    dataset = BalancedASVspoofLADataset(la_path if os.path.exists(la_path) else DATA_DIR, max_per_class=500)
    
    if len(dataset) == 0:
        print("[Warning] No local evaluation dataset found.")
        return

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []
    y_prob = []

    print(f"Running evaluation inference across {len(dataset)} balanced audio samples...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', labels=[0, 1])

    print("\n" + "=" * 65)
    print("  MODEL 2 (AUDIO SYNTHETIC VOICE DETECTOR) RESEARCH EVALUATION METRICS  ")
    print("=" * 65)
    print(f" Dataset Name          : ASVspoof 2019 (Logical Access - LA Track)")
    print(f" Balanced Test Samples : {len(y_true):,}")
    print("-" * 65)
    print(f" Overall Test Accuracy : {acc * 100:.4f}%")
    print(f" Precision (Spoof/Fake): {prec * 100:.4f}%")
    print(f" Recall (Spoof/Fake)   : {rec * 100:.4f}%")
    print(f" F1-Score (Spoof/Fake) : {f1 * 100:.4f}%")
    print("-" * 65)
    print("\nFULL CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Genuine Human (0)", "Synthetic Spoof (1)"], digits=4))
    print("=" * 65)

    # 1. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Human", "AI Spoof"], yticklabels=["Human", "AI Spoof"], annot_kws={"size": 14})
    plt.title("Confusion Matrix - Audio Synthetic Voice CNN", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("True Label", fontsize=12, labelpad=8)
    plt.tight_layout()
    cm_path = os.path.join(PAPER_DIR, "audio_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Audio Confusion Matrix Plot to: {cm_path}")

    # 2. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="forestgreen", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - PyTorch Audio Spectrogram CNN", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    roc_path = os.path.join(PAPER_DIR, "audio_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved Audio ROC Curve Plot to: {roc_path}")


if __name__ == "__main__":
    generate_audio_research_figures()
