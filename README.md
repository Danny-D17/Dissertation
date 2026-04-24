# COVID-19 Detection from CT Scans using Deep Learning

A deep learning system for detecting COVID-19 from chest CT scan images, developed as a final year dissertation project. The system trains and evaluates two models — a custom CNN and a DenseNet121 transfer learning model — on the MosMedData dataset, and serves predictions through a Flask web application with Grad-CAM visualisation.

---

## Project Structure

```
DISSERTATION/
│
├── app.py                               # Flask application entry point
│
├── data_slices/                         # Preprocessed CT slices (11,100 total)
│   ├── CT-0/                            # No-COVID slices
│   ├── CT-1/                            # Mild COVID slices
│   ├── CT-2/                            # Moderate COVID slices
│   ├── CT-3/                            # Severe COVID slices
│   └── CT-4/                            # Critical COVID slices
│
├── gradcam_binary/                      # Grad-CAM output images (custom CNN)
├── gradcam_densenet121/                 # Grad-CAM output images (DenseNet121)
│
├── models/
│   ├── mosmed_cnn_baseline.keras        # Initial CNN baseline model
│   ├── mosmed_cnn_binary.keras          # Final custom CNN model
│   └── mosmed_densenet121_binary.keras  # Final DenseNet121 model
│
├── MosMedData Chest CT Scans with COVID-19/  # Raw NIfTI dataset
│
├── scripts/
│   ├── evaluate_cnn_binary.py           # Custom CNN evaluation and metrics
│   ├── evaluate_cnn.py                  # Initial CNN evaluation
│   ├── evaluate_densenet121_binary.py   # DenseNet121 evaluation and metrics
│   ├── explore_mosmed.py                # Dataset exploration
│   ├── grad_cam_binary.py               # Grad-CAM for custom CNN
│   ├── grad_cam_densenet121_binary.py   # Grad-CAM for DenseNet121
│   ├── prepare_slices.py                # Slice extraction and preprocessing
│   ├── train_cnn_binary.py              # Final custom CNN training script
│   ├── train_cnn.py                     # Initial CNN training script
│   └── train_densenet121_binary.py      # DenseNet121 training script
│
├── static/
│   ├── gradcam/                         # Flask-served Grad-CAM overlays
│   └── uploads/                         # Temporarily stored uploaded slices
│
├── templates/
│   └── index.html                       # Upload and results interface
│
├── .gitattributes
└── .gitignore
```

---

## Dataset

This project uses the [MosMedData: Chest CT Scans with COVID-19 Related Findings](https://mosmed.ai/en/datasets/datasets/covid191110/) dataset, which contains anonymised chest CT scans labelled across five severity categories (CT-0 to CT-4).

CT slices are extracted from NIfTI volumes using `scripts/prepare_slices.py` and saved as PNG images under `data_slices/`.

> **Note:** The raw dataset NIfTI files are not included in this repository due to size. Download from the MosMedData website and place in the `MosMedData Chest CT Scans with COVID-19/` directory.

---

## Models

| Model | File | Description |
|-------|------|-------------|
| CNN Baseline | `mosmed_cnn_baseline.keras` | Initial lightweight CNN |
| Custom CNN | `mosmed_cnn_binary.keras` | Optimised binary classification CNN |
| DenseNet121 | `mosmed_densenet121_binary.keras` | Transfer learning with ImageNet weights |

> **Note:** Trained `.keras` model files are not included in this repository due to file size. Run the training scripts to reproduce them.

---

## Setup

### Requirements

- Python 3.9+
- TensorFlow / Keras
- OpenCV
- Flask
- NumPy, Matplotlib, scikit-learn

### Install dependencies

```bash
pip install tensorflow opencv-python flask numpy matplotlib scikit-learn
```

### Prepare the dataset

```bash
python scripts/prepare_slices.py
```

### Train the models

```bash
# Custom CNN
python scripts/train_cnn_binary.py

# DenseNet121
python scripts/train_densenet121_binary.py
```

### Evaluate the models

```bash
python scripts/evaluate_cnn_binary.py
python scripts/evaluate_densenet121_binary.py
```

### Generate Grad-CAM visualisations

```bash
python scripts/grad_cam_binary.py
python scripts/grad_cam_densenet121_binary.py
```

---

## Flask Web Application

The web app allows users to upload a CT scan slice and receive a COVID-19 prediction from both models, along with a Grad-CAM heatmap overlay highlighting the regions influencing the prediction.

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to generate heatmaps that highlight which regions of a CT scan contributed most to the model's prediction. Output images are saved in `gradcam_binary/` and `gradcam_densenet121/`.

---

## Acknowledgements

- Dataset: [MosMedData](https://mosmed.ai/en/datasets/datasets/covid191110/)
- Supervisor: Oluseyi Oyedeji
- Built as part of a BSc/MSc dissertation at the University of Northampton

---

## License

This project is for academic purposes only.
