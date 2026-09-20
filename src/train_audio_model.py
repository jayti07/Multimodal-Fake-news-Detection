"""
Training Script for Audio Synthetic Voice Spoof Detector
Robust automatic ASVspoof 2019 LA protocol parser (Balanced Bonafide & Spoof) & PyTorch 2D CNN trainer.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def extract_mel_spectrogram(audio_path: str, n_mels=128, max_len=128):
    """
    Loads an audio clip (.flac / .wav / .mp3) and converts it into a (1, 128, 128) Mel-Spectrogram matrix.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3.0)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] < max_len:
            pad_width = max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_len]

        mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        return torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 128, 128)
    except Exception as e:
        print(f"[Audio Error] Could not process {audio_path}: {e}")
        return None


class ASVspoofLADataset(Dataset):
    """
    Direct Balanced Dataset Loader for ASVspoof 2019 LA directory.
    Searches for countermeasure (.cm.) protocol files and maps bonafide (0) vs spoof (1) in balanced numbers.
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

        if not protocol_file:
            for root, dirs, files in os.walk(la_dir_path):
                for f in files:
                    if f.endswith(".txt") and ("train" in f or "dev" in f or "eval" in f):
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
            print(f"[ASVspoof Loader] Found Protocol: {protocol_file}")
            print(f"[ASVspoof Loader] Found Audio Dir: {audio_dir}")
            
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

            print(f"[ASVspoof Loader] Balanced Load Complete: {real_count} Bonafide (0) + {fake_count} Spoof (1) = {len(self.samples)} total samples")

        if len(self.samples) == 0:
            real_dir = os.path.join(DATA_DIR, "real")
            fake_dir = os.path.join(DATA_DIR, "fake")
            if os.path.exists(real_dir):
                for fname in os.listdir(real_dir):
                    if fname.lower().endswith(('.wav', '.mp3', '.flac')):
                        self.samples.append((os.path.join(real_dir, fname), 0))
            if os.path.exists(fake_dir):
                for fname in os.listdir(fake_dir):
                    if fname.lower().endswith(('.wav', '.mp3', '.flac')):
                        self.samples.append((os.path.join(fake_dir, fname), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        tensor_spec = extract_mel_spectrogram(audio_path)
        if tensor_spec is None:
            tensor_spec = torch.zeros((1, 128, 128), dtype=torch.float32)
        return tensor_spec, torch.tensor(label, dtype=torch.long)


class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super(SimpleAudioCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_audio_model(la_dir_path=None, epochs=5, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    dataset = ASVspoofLADataset(la_dir_path or DATA_DIR, max_per_class=500)

    if len(dataset) == 0:
        print(f"\n[Error] No audio files found!")
        print(f"Please check your path: {la_dir_path}")
        return

    print(f"\nSuccessfully loaded {len(dataset)} balanced audio clips for training!")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleAudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n--- Training Audio Spectrogram PyTorch CNN ---")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "audio_spoof_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nSuccessfully saved trained Audio CNN model to: {save_path}")


if __name__ == "__main__":
    la_path = sys.argv[1] if len(sys.argv) > 1 else None
    train_audio_model(la_dir_path=la_path)
