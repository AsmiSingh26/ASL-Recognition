import os
import json
import tensorflow as tf
from src.model import build_dual_input_model
from src.dataset import ASLDataGenerator

DATA_DIR = "data/train"
VAL_DIR  = "data/test"
BATCH_SIZE = 16
EPOCHS = 25
MODEL_SAVE_PATH = "models/asl_dual_model.h5"
LABEL_MAP_PATH  = "models/label_map.json"

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

classes = sorted(os.listdir(DATA_DIR))
label_map = {cls: idx for idx, cls in enumerate(classes)}

with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f)
print("Classes:", label_map)
print("Total classes:", len(label_map))

train_gen = ASLDataGenerator(DATA_DIR, label_map, BATCH_SIZE, augment=True)
val_gen   = ASLDataGenerator(VAL_DIR,  label_map, BATCH_SIZE, augment=False)

model = build_dual_input_model(num_classes=len(label_map))
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("Training complete! Model saved to", MODEL_SAVE_PATH)