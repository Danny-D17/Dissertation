import tensorflow as tf
import os

DATA_DIR = "/Users/daniel/Dissertation/data_slices"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123

# ---- 1. Load 5-class dataset as before ----
train_5 = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)

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

print("Original classes:", train_5.class_names)  # ['CT-0', 'CT-1', 'CT-2', 'CT-3', 'CT-4']

# ---- 2. Map 5 labels -> binary labels ----
# 0 stays 0 (no findings), 1–4 become 1 (any COVID pneumonia/severity)
def to_binary(x, y):
    y_bin = tf.where(y == 0, 0, 1)
    return x, tf.cast(y_bin, tf.float32)

train_bin = train_5.map(to_binary)
val_bin = val_5.map(to_binary)

AUTOTUNE = tf.data.AUTOTUNE
train_bin = train_bin.cache().shuffle(1000).prefetch(AUTOTUNE)
val_bin = val_bin.cache().prefetch(AUTOTUNE)

# ---- 3. Define binary CNN model ----
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),  # binary output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

# ---- 4. Train ----
history = model.fit(
    train_bin,
    validation_data=val_bin,
    epochs=10,
)

# ---- 5. Save binary model in multiple formats ----
os.makedirs("/Users/daniel/Dissertation/models", exist_ok=True)

# Save as SavedModel format (just use directory path without extension)
model.save("/Users/daniel/Dissertation/models/mosmed_cnn_binary_savedmodel")

# Also save as .keras format for backup
model.save("/Users/daniel/Dissertation/models/mosmed_cnn_binary.keras")

# Save as .h5 format
model.save("/Users/daniel/Dissertation/models/mosmed_cnn_binary.h5")

print("\n" + "="*50)
print("Model saved successfully!")
print("SavedModel: models/mosmed_cnn_binary_savedmodel/")
print("Keras format: models/mosmed_cnn_binary.keras")
print("H5 format: models/mosmed_cnn_binary.h5")
print("="*50)


