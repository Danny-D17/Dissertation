import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# --------------------
# Config
# --------------------
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'
MODEL_PATH = 'models/mosmed_densenet121_binary.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# --------------------
# Load model once at startup
# --------------------
print("Loading DenseNet121 model...")
model = tf.keras.models.load_model(MODEL_PATH)
preprocess_input = tf.keras.applications.densenet.preprocess_input
print("✓ Model loaded successfully")

# --------------------
# Helper functions
# --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(image_path):
    """
    Matches your exact preprocessing pipeline from grad_cam_densenet121_binary.py
    """
    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 224x224
    img = cv2.resize(img, IMG_SIZE)
    # Convert to float32 and normalize to [0, 255] range
    img = img.astype('float32')
    # Convert grayscale to RGB by stacking 3 channels
    img_rgb = np.stack([img, img, img], axis=-1)
    return img_rgb

def make_gradcam_heatmap(img_array):
    """
    Your exact M1-proof Grad-CAM function from grad_cam_densenet121_binary.py
    """
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        # Apply DenseNet preprocessing
        x = preprocess_input(img_tensor)
        
        # Forward pass through DenseNet121 backbone
        densenet_layer = model.get_layer("densenet121")
        conv_outputs = densenet_layer(x, training=False)
        tape.watch(conv_outputs)
        
        # Manually pass through head layers
        x = model.get_layer("global_average_pooling2d")(conv_outputs)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x, training=False)
        predictions = model.get_layer("dense_1")(x)
        
        # Get COVID class score
        target_class_score = predictions[:, 0]
    
    # Compute gradients
    grads = tape.gradient(target_class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv outputs by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU + Normalize
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape)
    heatmap = (heatmap / max_val).numpy()
    
    return heatmap

def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    """
    Your exact overlay function from grad_cam_densenet121_binary.py
    """
    img_gray = image[:, :, 0]  # Use first channel
    img_norm = img_gray / 255.0 if img_gray.max() > 1.0 else img_gray
    H, W = img_norm.shape[:2]
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    
    # Blend
    overlay = heatmap_color * alpha + np.stack([img_norm]*3, axis=-1)
    return np.clip(overlay, 0, 1)

# --------------------
# Flask routes
# --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    
    # POST: handle file upload
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type. Please upload PNG, JPG, or JPEG')
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    try:
        # Preprocess image (your exact pipeline)
        img_rgb = load_and_preprocess_image(upload_path)
        img_batch = np.expand_dims(img_rgb, axis=0)
        
        # Get prediction
        prob = float(model.predict(img_batch, verbose=0)[0][0])
        prediction = 'COVID' if prob >= 0.5 else 'no_COVID'
        
        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_batch)
        overlay = overlay_heatmap_on_image(img_rgb, heatmap)
        
        # Save Grad-CAM overlay
        gradcam_filename = f'gradcam_{filename}'
        gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
        overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(gradcam_path, overlay_bgr)
        
        # Generate URLs for display
        image_url = url_for('static', filename=f'uploads/{filename}')
        gradcam_url = url_for('static', filename=f'gradcam/{gradcam_filename}')
        
        return render_template('index.html',
                             prediction=prediction,
                             probability=round(prob * 100, 2),
                             image_url=image_url,
                             gradcam_url=gradcam_url)
    
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)