import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── Load data ──
SAVE_DIR = "data/extracted"
MODEL_PATH = "models/asl_landmark_model.h5"
MAP_PATH = "models/label_map.json"

X = np.load(f"{SAVE_DIR}/X.npy")
y = np.load(f"{SAVE_DIR}/y.npy")

with open(MAP_PATH) as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}
class_names = [idx_to_label[i] for i in range(len(label_map))]

# ── Filter zero vectors ──
mask = ~np.all(X == 0, axis=1)
X = X[mask]
y = y[mask]
print(f"Loaded {len(X)} clean samples")

# ── Split (same seed as training) ──
_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Validation samples: {len(X_val)}")

# ── Load model & predict ──
model = tf.keras.models.load_model(MODEL_PATH)
y_pred_probs = model.predict(X_val, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# ── Confusion Matrix ──
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(18, 16))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5
)
plt.title('ASL Recognition — Confusion Matrix\n(Val Accuracy: 99.14%)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix.png")

# ── Per-class accuracy ──
print("\n" + "="*50)
print("PER-CLASS ACCURACY")
print("="*50)
per_class = cm.diagonal() / cm.sum(axis=1) * 100
for i, (cls, acc) in enumerate(zip(class_names, per_class)):
    bar = '█' * int(acc // 5) + '░' * (20 - int(acc // 5))
    print(f"{cls:8} {bar} {acc:.1f}%")

# ── Classification Report ──
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_val, y_pred, target_names=class_names))

# ── Find worst performing classes ──
print("="*50)
print("BOTTOM 5 CLASSES (need improvement)")
print("="*50)
worst = sorted(zip(class_names, per_class), key=lambda x: x[1])[:5]
for cls, acc in worst:
    print(f"  {cls}: {acc:.1f}%")

print("\nTOP 5 CLASSES (best performance)")
print("="*50)
best = sorted(zip(class_names, per_class), key=lambda x: x[1], reverse=True)[:5]
for cls, acc in best:
    print(f"  {cls}: {acc:.1f}%")