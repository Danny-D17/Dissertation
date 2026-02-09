import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# --------------------
# Config
# --------------------
DATA_DIR = "/Users/daniel/Dissertation/data_slices"
MODEL_PATH = "/Users/daniel/Dissertation/models/mosmed_cnn_binary.keras"
OUTPUT_DIR = "/Users/daniel/Dissertation/gradcam_binary"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# 1. Rebuild validation dataset (same as eval script)
# --------------------
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

def to_binary(x, y):
    y_bin = tf.where(y == 0, 0, 1)
    return x, tf.cast(y_bin, tf.float32)

val_bin = val_5.map(to_binary)

AUTOTUNE = tf.data.AUTOTUNE
val_bin = val_bin.cache().prefetch(AUTOTUNE)

# Turn into numpy arrays for easier indexing
images_list = []
labels_list = []
for x_batch, y_batch in val_bin:
    images_list.append(x_batch.numpy())
    labels_list.append(y_batch.numpy())

val_images = np.concatenate(images_list, axis=0)   # (2220, 256, 256, 1)
val_labels = np.concatenate(labels_list, axis=0)   # (2220, 1)

# --------------------
# 2. Load model and identify last conv + classifier
# --------------------
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# Last conv layer name
LAST_CONV_LAYER_NAME = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        LAST_CONV_LAYER_NAME = layer.name
        break

if LAST_CONV_LAYER_NAME is None:
    raise ValueError("No Conv2D layer found in model.")

print("Using last conv layer:", LAST_CONV_LAYER_NAME)

last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)

# Split model into: base_model (up to last conv) and classifier
# Following the Keras Grad-CAM example [web:181]
# base_model: input -> last_conv_layer output
base_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=last_conv_layer.output,
)

# classifier_model: takes last_conv output, passes through the remaining layers
# Find the index of the last_conv_layer in model.layers
last_conv_index = None
for i, layer in enumerate(model.layers):
    if layer.name == LAST_CONV_LAYER_NAME:
        last_conv_index = i
        break

# Remaining layers after last conv
classifier_layers = model.layers[last_conv_index + 1:]

# Build a small Sequential classifier on top of conv feature maps
classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer in classifier_layers:
    x = layer(x)
classifier_output = x
classifier_model = tf.keras.models.Model(
    inputs=classifier_input,
    outputs=classifier_output,
)

# --------------------
# 3. Grad-CAM function (Keras pattern)
# --------------------
def make_gradcam_heatmap(img_array):
    """
    img_array: shape (1, H, W, 1)
    Returns: heatmap (Hc, Wc) in [0,1]
    """
    # 1. Forward pass through base_model
    conv_outputs = base_model(img_array)

    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        # 2. Forward pass through classifier on conv outputs
        preds = classifier_model(conv_outputs)
        # Binary sigmoid scalar output
        target = preds[:, 0]

    # 3. Compute gradients of the target with respect to conv_outputs
    grads = tape.gradient(target, conv_outputs)  # (1, Hc, Wc, C)
    if grads is None:
        raise RuntimeError("Gradients are None – cannot compute Grad-CAM.")

    # 4. Global average pooling on gradients -> channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]  # (Hc, Wc, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (Hc, Wc)

    # ReLU + normalize to [0,1]
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return tf.zeros_like(heatmap).numpy()
    heatmap /= max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    """
    image: (H, W) or (H, W, 1) in [0, 255] or [0,1]
    heatmap: (Hc, Wc) in [0,1]
    """
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    img_norm = image.astype("float32")
    if img_norm.max() > 1.0:
        img_norm /= 255.0

    H, W = img_norm.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    overlay = heatmap_color * alpha + np.stack([img_norm]*3, axis=-1)
    overlay = np.clip(overlay, 0, 1)
    return overlay

# --------------------
# 4. Pick examples (TP, FP, TN, FN) and save
# --------------------
probs = model.predict(val_images, verbose=0).ravel()
preds = (probs >= 0.5).astype(int)
true = val_labels.astype(int).ravel()

indices_tp = np.where((true == 1) & (preds == 1))[0]
indices_fn = np.where((true == 1) & (preds == 0))[0]
indices_tn = np.where((true == 0) & (preds == 0))[0]
indices_fp = np.where((true == 0) & (preds == 1))[0]

print("TP:", len(indices_tp), "FN:", len(indices_fn),
      "TN:", len(indices_tn), "FP:", len(indices_fp))

def save_example(index, label_str, pred_str):
    img = val_images[index]  # (256,256,1)
    x = np.expand_dims(img, axis=0)  # (1,256,256,1)

    heatmap = make_gradcam_heatmap(x)
    overlay = overlay_heatmap_on_image(img[..., 0], heatmap)

    prob = probs[index]
    fname = f"{label_str}_pred-{pred_str}_idx-{index}_prob-{prob:.3f}.png"
    path = os.path.join(OUTPUT_DIR, fname)

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(overlay)
    plt.title(f"True: {label_str}, Pred: {pred_str}, p(COVID)={prob:.3f}")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

for i in indices_tp[:3]:
    save_example(i, "COVID", "COVID")

for i in indices_fn[:3]:
    save_example(i, "COVID", "no_COVID")

for i in indices_tn[:3]:
    save_example(i, "no_COVID", "no_COVID")

for i in indices_fp[:3]:
    save_example(i, "no_COVID", "COVID")

