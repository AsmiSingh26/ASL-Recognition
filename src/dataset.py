import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.utils import Sequence
from src.preprocessing import normalize_landmarks, preprocess_image, augment_advanced

class ASLDataGenerator(Sequence):
    def __init__(self, data_dir, label_map, batch_size=32, augment=False):
        self.data_dir = data_dir
        self.label_map = label_map
        self.batch_size = batch_size
        self.augment = augment
        self.samples = self._load_samples()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _load_samples(self):
        samples = []
        for label, idx in self.label_map.items():
            folder = os.path.join(self.data_dir, label)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    samples.append((os.path.join(folder, fname), idx))
        np.random.shuffle(samples)
        return samples

    def __len__(self):
        return max(1, len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, landmarks_batch, labels = [], [], []

        for img_path, label in batch:
            img = cv2.imread(img_path)
            if img is None:
                continue
            if self.augment:
                img = augment_advanced(img)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks:
                pts = []
                h, w, _ = img.shape
                for lm in result.multi_hand_landmarks[0].landmark:
                    pts.append((int(lm.x * w), int(lm.y * h), lm.z))
                landmark_vec = normalize_landmarks(pts)
            else:
                landmark_vec = np.zeros(63)

            processed_img = preprocess_image(img)
            images.append(processed_img)
            landmarks_batch.append(landmark_vec)
            labels.append(label)

        return (
            {
                "image_input": np.array(images),
                "landmark_input": np.array(landmarks_batch)
            },
            np.array(labels)
        )