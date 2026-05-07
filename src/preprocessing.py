import numpy as np
import cv2
import albumentations as A

def normalize_landmarks(landmarks):
    pts = np.array(landmarks, dtype=np.float32)
    pts = pts - pts[0]
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts.flatten()

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(scale=(0.9, 1.1), p=0.5)
])

def augment_advanced(img):
    augmented = augmentor(image=img)
    return augmented["image"]