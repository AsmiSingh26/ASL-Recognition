import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

class HandTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_hand_data(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, "No hand detected"

        h, w, _ = frame.shape
        points = []

        for lm in result.multi_hand_landmarks[0].landmark:
            x, y, z = int(lm.x * w), int(lm.y * h), lm.z
            points.append((x, y, z))

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        pad = 20
        x_min = max(0, min(x_coords) - pad)
        y_min = max(0, min(y_coords) - pad)
        x_max = min(w, max(x_coords) + pad)
        y_max = min(h, max(y_coords) + pad)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None, "Invalid ROI"

        roi_resized = cv2.resize(roi, (224, 224))

        return {
            "landmarks": points,
            "roi": roi_resized,
            "bbox": (x_min, y_min, x_max, y_max)
        }, None