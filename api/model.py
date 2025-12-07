"""
============================================================
MODEL - Face Detection & Classification
============================================================
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from facenet_pytorch import MTCNN
import cv2
import os

from .config import (
    CLASS_NAMES, ROLE_MAPPING, 
    CONFIDENCE_THRESHOLD, FACE_DETECTION_THRESHOLD,
    MODEL_PATH
)
from .database import log_access


class FaceRecognitionModel:
    """Face Recognition Model using MTCNN + ViT"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = None
        self.mtcnn = None
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """Initialize model and MTCNN"""
        if self._initialized:
            return
        
        print("=" * 60)
        print("      INITIALIZING FACE RECOGNITION MODEL")
        print("=" * 60)
        print(f"✅ Device: {self.device}")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # MTCNN
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.6],
            factor=0.709,
            post_process=False
        )
        print("✅ MTCNN initialized")
        
        # ViT Model
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(CLASS_NAMES),
            ignore_mismatched_sizes=True
        )
        
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model ready")
        print("=" * 60)
        
        self._initialized = True
    
    @staticmethod
    def get_full_label(name):
        """Get full label with role and authorization status"""
        name_lower = name.lower()
        if name_lower in ROLE_MAPPING:
            role = ROLE_MAPPING[name_lower]
            return f"{name.capitalize()} ({role})", role, True
        else:
            return f"{name} (Guest)", "Guest", False
    
    def process_image(self, image, save_log=True):
        """Process image and return detection results"""
        if not self._initialized:
            self.initialize()
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(pil_image)
        
        results = []
        
        if boxes is None:
            return results
        
        for box, prob in zip(boxes, probs):
            if prob is None or prob < FACE_DETECTION_THRESHOLD:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding
            w, h = x2 - x1, y2 - y1
            pad = int(max(w, h) * 0.15)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(pil_image.size[0], x2 + pad)
            y2 = min(pil_image.size[1], y2 + pad)
            
            # Crop and classify
            face = pil_image.crop((x1, y1, x2, y2))
            face_tensor = self.transform(face).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(face_tensor).logits
                probs_cls = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs_cls, 1)
            
            confidence = confidence.item()
            
            if confidence >= CONFIDENCE_THRESHOLD:
                name = CLASS_NAMES[predicted.item()]
                full_label, role, authorized = self.get_full_label(name)
            else:
                name = "Unknown"
                full_label = "Unknown (Guest)"
                role = "Guest"
                authorized = False
            
            result = {
                "name": name.capitalize(),
                "role": role,
                "full_label": full_label,
                "authorized": authorized,
                "confidence": round(confidence * 100, 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "detection_score": round(float(prob), 2)
            }
            results.append(result)
            
            # Log to database
            if save_log:
                log_access(name, role, authorized, confidence)
        
        return results
    
    def process_frame(self, frame):
        """Process video frame and return annotated frame"""
        if not self._initialized:
            self.initialize()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        boxes, probs = self.mtcnn.detect(pil_image)
        
        results = []
        
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is None or prob < FACE_DETECTION_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Add padding
                w, h = x2 - x1, y2 - y1
                pad = int(max(w, h) * 0.15)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                
                # Classify
                face = pil_image.crop((x1, y1, x2, y2))
                face_tensor = self.transform(face).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(face_tensor).logits
                    probs_cls = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs_cls, 1)
                
                confidence = confidence.item()
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    name = CLASS_NAMES[predicted.item()]
                    full_label, role, authorized = self.get_full_label(name)
                else:
                    full_label = "Unknown (Guest)"
                    authorized = False
                
                # Draw
                color = (0, 255, 0) if authorized else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{full_label} ({confidence*100:.1f}%)"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                results.append({
                    "full_label": full_label,
                    "authorized": authorized,
                    "confidence": confidence
                })
        
        return frame, results


# Singleton instance
face_model = FaceRecognitionModel()
