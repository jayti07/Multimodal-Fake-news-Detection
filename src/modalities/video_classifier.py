"""
Video Deepfake Detection Module
Extracts video frames using OpenCV, crops face bounding boxes using MTCNN (facenet-pytorch),
and predicts per-frame deepfake probabilities using EfficientNet-B0.
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Any, List

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))


class VideoDeepfakeClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(MODEL_DIR, "video_deepfake_efficientnet.pth")
        
        # Load Pretrained EfficientNet-B0 backbone
        self.model = models.efficientnet_b0(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 2)  # Class 0: Real, Class 1: Deepfake

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                print("[VideoClassifier] Loaded EfficientNet Deepfake model weights.")
            except Exception as e:
                print(f"[VideoClassifier Warning] Could not load video weights: {e}")

        # Image Preprocessing Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # MTCNN Face Detector
        if HAS_MTCNN:
            self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)
        else:
            self.mtcnn = None

    def extract_face_crops(self, video_path: str, max_frames: int = 10) -> List[Image.Image]:
        """
        Extracts up to max_frames face crops from a video clip.
        """
        face_crops = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[VideoClassifier Error] Cannot open video file: {video_path}")
            return face_crops

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return face_crops

        step = max(1, total_frames // (max_frames + 1))
        frame_idx = step

        while cap.isOpened() and len(face_crops) < max_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(frame_rgb)

            if self.mtcnn is not None:
                try:
                    boxes, _ = self.mtcnn.detect(face_pil)
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
                            face_crops.append(crop_img)
                        else:
                            face_crops.append(face_pil.resize((224, 224)))
                    else:
                        face_crops.append(face_pil.resize((224, 224)))
                except Exception:
                    face_crops.append(face_pil.resize((224, 224)))
            else:
                face_crops.append(face_pil.resize((224, 224)))

            frame_idx += step

        cap.release()
        return face_crops

    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Runs per-frame face deepfake prediction and aggregates results.
        """
        if not video_path or not os.path.exists(video_path):
            return {"label": "Unknown", "confidence": 0.0, "error": "Video file not found"}

        face_crops = self.extract_face_crops(video_path, max_frames=10)

        if not face_crops:
            return {
                "label": "No Face Detected",
                "confidence": 0.0,
                "error": "No human faces were detected in the uploaded video.",
                "model_used": "AI Facial Detection System"
            }

        scores = []
        with torch.no_grad():
            for face_pil in face_crops:
                img_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                scores.append(float(probs[1]))  # Probability of Deepfake

        avg_deepfake_prob = sum(scores) / len(scores) if len(scores) > 0 else 0.5
        label = "Manipulated / Deepfake Video" if avg_deepfake_prob > 0.5 else "Authentic / Real Video"
        confidence = round(max(avg_deepfake_prob, 1.0 - avg_deepfake_prob) * 100, 2)

        return {
            "label": label,
            "confidence": confidence,
            "deepfake_probability": round(avg_deepfake_prob, 4),
            "faces_analyzed": len(face_crops),
            "model_used": "AI Visual Face Analysis System"
        }

