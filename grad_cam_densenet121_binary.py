import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# --------------------
# Config
# --------------------
DATA_DIR = "/Users/daniel/Dissertation/data_slices"
MODEL_PATH = "/Users/daniel/Dissertation/models/mosmed_densenet121_binary.keras"
OUTPUT_DIR = "/Users/daniel/Dissertation/gradcam_densenet121"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# 1. Rebuild validation dataset
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

def to_binary_and_rgb(x, y):
    y_bin = tf.where(y == 0, 0, 1)
    x_rgb = tf.image.grayscale_to_rgb(x)
    return x_rgb, tf.cast(y_bin, tf.float32)

val_bin = val_5.map(to_binary_and_rgb).cache().prefetch(tf.data.AUTOTUNE)

# Turn into numpy arrays
images_list, labels_list = [], []
for x_batch, y_batch in val_bin:
    images_list.append(x_batch.numpy())
    labels_list.append(y_batch.numpy())

val_images = np.concatenate(images_list, axis=0)
val_labels = np.concatenate(labels_list, axis=0)

# --------------------
# 2. Load model and Preprocessing Function
# --------------------
model = tf.keras.models.load_model(MODEL_PATH)
preprocess_input = tf.keras.applications.densenet.preprocess_input

# --------------------
# 3. The "M1-Proof" Grad-CAM function
# --------------------
def make_gradcam_heatmap(img_array):
    """
    Directly calls the model components to bypass graph tracing errors.
    """
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        # 1. Apply the same preprocessing used in training
        x = preprocess_input(img_tensor)
        
        # 2. Forward pass through the DenseNet121 backbone
        densenet_layer = model.get_layer("densenet121")
        conv_outputs = densenet_layer(x, training=False)
        tape.watch(conv_outputs)
        
        # 3. Manually pass through the remaining head layers
        # This bypasses the Functional.call() KeyError
        x = model.get_layer("global_average_pooling2d")(conv_outputs)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x, training=False)
        predictions = model.get_layer("dense_1")(x)
        
        # 4. Get the COVID class score
        target_class_score = predictions[:, 0]

    # Compute gradients of the prediction w.r.t. the conv feature maps
    grads = tape.gradient(target_class_score, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv output channels by their importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU + Normalize to [0, 1]
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape)
    
    heatmap = (heatmap / max_val).numpy()
    return heatmap

def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    img_gray = image[:, :, 0] # Use first channel of the RGB image
    img_norm = (img_gray / 255.0) if img_gray.max() > 1.0 else img_gray
    
    H, W = img_norm.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))
    
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    
    overlay = heatmap_color * alpha + np.stack([img_norm]*3, axis=-1)
    return np.clip(overlay, 0, 1)

# --------------------
# 4. Execute and Save
# --------------------
print("Computing predictions and generating Grad-CAM...")
probs = model.predict(val_images, verbose=0).ravel()
preds = (probs >= 0.5).astype(int)
true = val_labels.astype(int).ravel()

indices_tp = np.where((true == 1) & (preds == 1))[0]
indices_fn = np.where((true == 1) & (preds == 0))[0]
indices_tn = np.where((true == 0) & (preds == 0))[0]
indices_fp = np.where((true == 0) & (preds == 1))[0]

def save_example(index, label_str, pred_str):
    img = val_images[index]
    x = np.expand_dims(img, axis=0)
    
    heatmap = make_gradcam_heatmap(x)
    overlay = overlay_heatmap_on_image(img, heatmap)
    
    prob = probs[index]
    fname = f"DenseNet_{label_str}_pred-{pred_str}_idx-{index}.png"
    path = os.path.join(OUTPUT_DIR, fname)
    
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(overlay)
    plt.title(f"DenseNet - True: {label_str}, Pred: {pred_str}\np(COVID)={prob:.3f}")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

for i in indices_tp[:3]: save_example(i, "COVID", "COVID")
for i in indices_fn[:3]: save_example(i, "COVID", "no_COVID")
for i in indices_tn[:3]: save_example(i, "no_COVID", "no_COVID")
for i in indices_fp[:3]: save_example(i, "no_COVID", "COVID")

print("\nSuccess! DenseNet121 Grad-CAM complete.")
