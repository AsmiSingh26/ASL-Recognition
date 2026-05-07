import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ── Load model & label map ──
model = tf.keras.models.load_model("models/asl_landmark_model.h5")

with open("models/label_map.json") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}

# ── MediaPipe ──
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def normalize(landmarks):
    pts = np.array(landmarks, dtype=np.float32)
    pts = pts - pts[0]
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0: pts = pts / d
    return pts.flatten()

# ── Smoothing buffer ──
buffer = deque(maxlen=5)
CONFIDENCE_THRESHOLD = 0.85

# ── Webcam ──
cap = cv2.VideoCapture(0)
print("Starting webcam — press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand, mp.solutions.hands.HAND_CONNECTIONS
        )

        # Extract & normalize
        pts = [(int(lm.x*w), int(lm.y*h), lm.z)
               for lm in hand.landmark]
        feat = normalize(pts).reshape(1, -1)

        # Predict
        probs = model.predict(feat, verbose=0)[0]
        top3_idx  = np.argsort(probs)[::-1][:3]
        top1_conf = probs[top3_idx[0]]
        top1_label = idx_to_label[top3_idx[0]]

        # Smooth
        buffer.append(top1_label)
        smoothed = max(set(buffer), key=list(buffer).count)

        # Status
        if top1_conf >= CONFIDENCE_THRESHOLD:
            status = "Confident"
            color  = (0, 255, 0)
        else:
            status = "Uncertain"
            color  = (0, 165, 255)
            smoothed = "?"

        # ── Overlay ──
        # Main prediction
        cv2.putText(frame, f"Letter: {smoothed}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

        # Confidence bar background
        cv2.rectangle(frame, (20, 90), (320, 115), (50, 50, 50), -1)
        bar_w = int(top1_conf * 300)
        cv2.rectangle(frame, (20, 90), (20 + bar_w, 115), color, -1)
        cv2.putText(frame, f"{top1_conf*100:.1f}%", (325, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Status
        cv2.putText(frame, status, (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Top 3
        cv2.putText(frame, "Top 3:", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for i, idx in enumerate(top3_idx):
            txt = f"{idx_to_label[idx]}: {probs[idx]*100:.1f}%"
            cv2.putText(frame, txt, (20, 210 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    else:
        cv2.putText(frame, "No hand detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        buffer.clear()

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done!")