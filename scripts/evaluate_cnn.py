import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = "/Users/daniel/Dissertation/data_slices"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123

# 1. Rebuild the validation dataset (same settings as training)
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

class_names = raw_val_ds.class_names
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
val_ds = raw_val_ds.cache().prefetch(AUTOTUNE)

# 2. Load the trained model
model_path = "/Users/daniel/Dissertation/models/mosmed_cnn_baseline.keras"
model = tf.keras.models.load_model(model_path)
print("Loaded model from:", model_path)

# 3. Overall loss and accuracy on validation set
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")

# 4. Get predictions and true labels for detailed metrics
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    preds_labels = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 5. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm)

# 6. Precision, recall, F1 per class
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))
