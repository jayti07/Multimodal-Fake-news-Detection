"""
Training Script for Video Deepfake EfficientNet Classifier (Celeb-DF v2 Benchmark)
Extracts face crops using facenet-pytorch (MTCNN) & OpenCV -> PyTorch EfficientNet-B0.
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "video_faces"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


class CelebDFDataset(Dataset):
    """
    Direct Video & Face Crop Dataset Loader for Celeb-DF v2 dataset.
    Scans Celeb-real (Label 0) and Celeb-synthesis (Label 1) folders.
    """
    def __init__(self, celeb_df_dir, max_videos_per_class=80, frames_per_video=3):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if HAS_MTCNN:
            mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
        else:
            mtcnn = None

        real_videos = []
        fake_videos = []

        # Search for Celeb-real and Celeb-synthesis directories
        for root, dirs, files in os.walk(celeb_df_dir):
            folder_name = os.path.basename(root).lower()
            if "celeb-real" in folder_name or "youtube-real" in folder_name:
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov')):
                        real_videos.append(os.path.join(root, f))
            elif "celeb-synthesis" in folder_name or "synthesis" in folder_name or "fake" in folder_name:
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov')):
                        fake_videos.append(os.path.join(root, f))

        print(f"[Celeb-DF Loader] Found {len(real_videos)} Real Videos and {len(fake_videos)} Deepfake Videos.")

        # Process Real Videos (Label 0)
        real_count = 0
        for vid_path in real_videos[:max_videos_per_class]:
            crops = self._extract_faces_from_video(vid_path, mtcnn, max_frames=frames_per_video)
            for crop in crops:
                self.samples.append((crop, 0))
            if crops:
                real_count += 1

        # Process Deepfake Videos (Label 1)
        fake_count = 0
        for vid_path in fake_videos[:max_videos_per_class]:
            crops = self._extract_faces_from_video(vid_path, mtcnn, max_frames=frames_per_video)
            for crop in crops:
                self.samples.append((crop, 1))
            if crops:
                fake_count += 1

        print(f"[Celeb-DF Loader] Extracted {len(self.samples)} total face crops ({real_count} Real vs {fake_count} Fake Videos)")

    def _extract_faces_from_video(self, video_path, mtcnn, max_frames=3):
        crops = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return crops

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return crops

        step = max(1, total_frames // (max_frames + 1))
        frame_idx = step

        while cap.isOpened() and len(crops) < max_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(frame_rgb)

            if mtcnn is not None:
                try:
                    # Detect face using MTCNN
                    boxes, _ = mtcnn.detect(face_pil)
                    if boxes is not None and len(boxes) > 0:
                        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                        w = x2 - x1
                        h = y2 - y1
                        margin = int(0.2 * w)
                        y1 = max(0, y1 - margin)
                        y2 = min(frame.shape[0], y2 + margin)
                        x1 = max(0, x1 - margin)
                        x2 = min(frame.shape[1], x2 + margin)

                        if x2 > x1 and y2 > y1:
                            crop_img = face_pil.crop((x1, y1, x2, y2))
                            crops.append(crop_img)
                        else:
                            crops.append(face_pil.resize((224, 224)))
                    else:
                        crops.append(face_pil.resize((224, 224)))
                except Exception:
                    crops.append(face_pil.resize((224, 224)))
            else:
                crops.append(face_pil.resize((224, 224)))

            frame_idx += step

        cap.release()
        return crops

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_pil, label = self.samples[idx]
        img_tensor = self.transform(face_pil)
        return img_tensor, torch.tensor(label, dtype=torch.long)


def train_video_model(celeb_df_path=None, epochs=5, batch_size=16, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    if not celeb_df_path or not os.path.exists(celeb_df_path):
        print(f"[Error] Please provide path to your Celeb-DF v2 dataset folder.")
        print(f"Usage: python src/train_video_model.py \"C:\\path\\to\\Celeb-DF-v2\"")
        return

    dataset = CelebDFDataset(celeb_df_path, max_videos_per_class=80, frames_per_video=3)

    if len(dataset) == 0:
        print(f"[Error] No face crops could be extracted from {celeb_df_path}")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load EfficientNet-B0 backbone
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # 0: Real, 1: Deepfake
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n--- Training PyTorch EfficientNet-B0 Video Deepfake Classifier ---")
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
    save_path = os.path.join(MODEL_DIR, "video_deepfake_efficientnet.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nSuccessfully saved trained Video EfficientNet model to: {save_path}")


if __name__ == "__main__":
    celeb_path = sys.argv[1] if len(sys.argv) > 1 else None
    train_video_model(celeb_df_path=celeb_path)
