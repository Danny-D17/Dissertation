import tensorflow as tf
import os

# --------------------
# Config
# --------------------
DATA_DIR = "/Users/daniel/Dissertation/data_slices"
MODEL_DIR = "/Users/daniel/Dissertation/models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE_BACKBONE = (224, 224)   # DenseNet121 input size
BATCH_SIZE = 32
SEED = 123
EPOCHS_HEAD = 5          
EPOCHS_FINETUNE = 10     

# --------------------
# 1. Build binary datasets (CT-0 vs CT-1..CT-4)
# --------------------
def make_binary_datasets():
    base_train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=IMG_SIZE_BACKBONE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=SEED,
    )

    base_val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=IMG_SIZE_BACKBONE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
    )

    print("Original classes:", base_train.class_names)

    def to_binary_and_rgb(x, y):
        # 0 -> no_COVID, 1–4 -> COVID
        y_bin = tf.where(y == 0, 0, 1)
        x_rgb = tf.image.grayscale_to_rgb(x)  # (H,W,1) -> (H,W,3)
        return x_rgb, tf.cast(y_bin, tf.float32)

    train_ds = base_train.map(to_binary_and_rgb)
    val_ds = base_val.map(to_binary_and_rgb)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds

train_ds, val_ds = make_binary_datasets()

# --------------------
# 2. Build DenseNet121 model
# --------------------
base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE_BACKBONE[0], IMG_SIZE_BACKBONE[1], 3),
)

# Use ImageNet preprocessing
preprocess_input = tf.keras.applications.densenet.preprocess_input

inputs = tf.keras.Input(shape=(IMG_SIZE_BACKBONE[0], IMG_SIZE_BACKBONE[1], 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)      
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs, name="densenet121_binary")

# --------------------
# 3. Phase 1 – train head only
# --------------------
base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
)

# --------------------
# 4. Phase 2 – fine-tune part of DenseNet121
# --------------------
# Unfreeze last N layers of backbone
fine_tune_at = len(base_model.layers) - 80  

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
)

# --------------------
# 5. Save model and basic eval
# --------------------
save_path = os.path.join(MODEL_DIR, "mosmed_densenet121_binary.keras")
model.save(save_path)
print("Saved DenseNet121 model to:", save_path)

val_loss, val_acc = model.evaluate(val_ds)
print(f"Final validation loss: {val_loss:.4f}")
print(f"Final validation accuracy: {val_acc:.4f}")
