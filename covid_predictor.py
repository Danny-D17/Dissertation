import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

def predict_covid(model_path, image_path):
    """
    Predict COVID-19 from CT scan image
    
    Args:
        model_path: Path to saved Keras model (.keras or .h5)
        image_path: Path to CT scan image
    """
    
    # Clear any previous TensorFlow session/graph
    tf.keras.backend.clear_session()
    
    # Load model fresh each time
    print(f"\nLoading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path)
    model_name = os.path.basename(model_path).lower()
    
    # Load and preprocess image
    print(f"Loading image: {os.path.basename(image_path)}")
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    img_array = np.array(img)
    
    # Preprocess based on model type
    if 'densenet' in model_name:
        # DenseNet: 224x224 RGB with ImageNet preprocessing
        img_resized = Image.fromarray(img_array).resize((224, 224))
        img_array = np.array(img_resized)
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
        img_batch = np.expand_dims(img_rgb, axis=0)
        img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_batch)
        print("Using DenseNet preprocessing (224x224 RGB)")
    else:
        # Custom CNN: 256x256 grayscale
        img_resized = Image.fromarray(img_array).resize((256, 256))
        img_array = np.array(img_resized)
        img_normalized = img_array.astype('float32') / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=-1)
        img_preprocessed = np.expand_dims(img_expanded, axis=0)
        print("Using Custom CNN preprocessing (256x256 grayscale)")
    
    # Predict - force eager execution to avoid caching
    print("\nPredicting...")
    prediction = float(model.predict(img_preprocessed, verbose=0)[0][0])
    
    # Interpret result
    if prediction >= 0.5:
        result = "COVID-19 POSITIVE"
        confidence = prediction * 100
    else:
        result = "COVID-19 NEGATIVE"
        confidence = (1 - prediction) * 100
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"p(COVID): {prediction:.4f}")
    print("="*50 + "\n")
    
    return prediction

if __name__ == "__main__":
    # Hardcoded paths - change these to test different images/models
    model_path = "/Users/daniel/Dissertation/models/mosmed_densenet121_binary.keras"
    image_path = "/Users/daniel/Dissertation/data_slices/CT-0/study_0001_slice000.png"  
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Run prediction
    predict_covid(model_path, image_path)
