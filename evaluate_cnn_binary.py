import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

DATA_DIR = "/Users/daniel/Dissertation/data_slices"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123

# 1. Rebuild 5-class validation dataset
val_5 = tf.keras.utils.image_dataset_from_directory(
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

print("Original classes:", val_5.class_names)

# 2. Map labels to binary: 0 -> 0, 1–4 -> 1
def to_binary(x, y):
    y_bin = tf.where(y == 0, 0, 1)
    return x, tf.cast(y_bin, tf.float32)

val_bin = val_5.map(to_binary)

AUTOTUNE = tf.data.AUTOTUNE
val_bin = val_bin.cache().prefetch(AUTOTUNE)

# 3. Load trained binary model
model_path = "/Users/daniel/Dissertation/models/mosmed_cnn_binary.keras" 

model = tf.keras.models.load_model(model_path)
print("Loaded model from:", model_path)

# 4. Overall loss and accuracy
val_loss, val_acc = model.evaluate(val_bin)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")

# 5. Collect predictions, true labels, and probabilities
y_true = []
y_pred = []
y_prob = []

for images, labels in val_bin:
    preds = model.predict(images, verbose=0).ravel()  # probabilities for COVID (class 1)

    y_true.extend(labels.numpy().astype(int))
    y_prob.extend(preds)
    y_pred.extend((preds >= 0.5).astype(int))  # threshold at 0.5

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# 6. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix (rows=true, cols=pred):")
# rows: [true 0, true 1], cols: [pred 0, pred 1]
print(cm)

# 7. Precision, recall, F1
print("\nClassification report (0 = CT-0, 1 = CT-1..CT-4):")
print(classification_report(y_true, y_pred, target_names=["no_COVID", "COVID"]))

# 8. ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_prob)  # positive class is 1 by default
roc_auc = auc(fpr, tpr)
print(f"\nROC AUC (COVID vs no_COVID): {roc_auc:.4f}")

# 9. Sensitivity and specificity at selected thresholds
def sens_spec_at(thresh):
    preds_t = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_t).ravel()
    sensitivity = tp / (tp + fn)  # recall for COVID
    specificity = tn / (tn + fp)  # recall for no_COVID
    return sensitivity, specificity

for thr in [0.3, 0.5, 0.7]:
    sens, spec = sens_spec_at(thr)
    print(f"\nThreshold {thr:.1f}:")
    print(f"  Sensitivity (COVID recall): {sens:.3f}")
    print(f"  Specificity (no_COVID recall): {spec:.3f}")
