import os, cv2, json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR   = "data/train"
SAVE_DIR   = "data/extracted"
MODEL_PATH = "models/asl_landmark_model.h5"
MAP_PATH   = "models/label_map.json"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def normalize(landmarks):
    pts = np.array(landmarks, dtype=np.float32)
    pts = pts - pts[0]
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0: pts = pts / d
    return pts.flatten()

classes   = sorted(os.listdir(DATA_DIR))
label_map = {cls: idx for idx, cls in enumerate(classes)}

X_path = os.path.join(SAVE_DIR, "X.npy")
y_path = os.path.join(SAVE_DIR, "y.npy")

if os.path.exists(X_path) and os.path.exists(y_path):
    print("Found saved features — skipping extraction!")
    X = np.load(X_path)
    y = np.load(y_path)
else:
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    all_X, all_y = [], []

    for cls, idx in label_map.items():
        folder = os.path.join(DATA_DIR, cls)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg','.png','.jpeg'))]
        print(f"Processing {cls} ({len(files)} images)...")

        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None: continue

            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                pts = []
                h, w, _ = img.shape
                for lm in result.multi_hand_landmarks[0].landmark:
                    pts.append((int(lm.x*w), int(lm.y*h), lm.z))
                feat = normalize(pts)
            else:
                feat = np.zeros(63)

            all_X.append(feat)
            all_y.append(idx)

    hands.close()
    X = np.array(all_X)
    y = np.array(all_y)
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"Saved {len(X)} samples!")

# ── Filter bad samples ──
print(f"\nBefore filter: {len(X)} samples")
mask = ~np.all(X == 0, axis=1)
X = X[mask]
y = y[mask]
print(f"After filter:  {len(X)} samples")
print(f"Removed:       {(~mask).sum()} bad samples\n")

with open(MAP_PATH, "w") as f:
    json.dump(label_map, f)

print(f"Training on {len(X)} samples, {len(classes)} classes")

# ── Split ──
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Bigger model ──
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, verbose=1
        )
    ]
)

model.save(MODEL_PATH)
print(f"\nDone! Model saved to {MODEL_PATH}")
print(f"Best val accuracy: {max(history.history['val_accuracy']):.2%}")