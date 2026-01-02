import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import nibabel as nib   # for .nii CT volumes

# EXACT folder name
BASE_DIR = "/Users/daniel/Dissertation/MosMedData Chest CT Scans with COVID-19 Related Findings COVID19_1110 1.0"
STUDIES_DIR = os.path.join(BASE_DIR, "studies")

class_name = "CT-0"   
class_dir = os.path.join(STUDIES_DIR, class_name)

print("class_dir:", class_dir)
print("Exists?", os.path.isdir(class_dir))
if not os.path.isdir(class_dir):
    raise SystemExit("ERROR: class_dir does not exist. Check BASE_DIR or folder name.")

# List study files (.nii)
study_ids = sorted([f for f in os.listdir(class_dir)
                    if f.lower().endswith(".nii")])
print(f"Found {len(study_ids)} studies in {class_name}")

first_study_path = os.path.join(class_dir, study_ids[0])
print("Using study file:", first_study_path)

# Load NIfTI volume
nii = nib.load(first_study_path)
volume = nii.get_fdata()          # shape: (H, W, slices) or similar
print("Volume shape:", volume.shape)

# Pick a few slice indices through the volume
num_slices = volume.shape[-1]
indices = np.linspace(0, num_slices - 1, 8, dtype=int)  # 8 slices across lung

images = [volume[:, :, i] for i in indices]

# Normalize to 0–255 for display
images = [(img - img.min()) / (img.max() - img.min() + 1e-8) * 255 for img in images]
images = [img.astype(np.uint8) for img in images]

# Show slices
cols = 4
rows = int(np.ceil(len(images) / cols))
plt.figure(figsize=(12, 3 * rows))

for i, img in enumerate(images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Slice {indices[i]}")
    plt.axis("off")

plt.suptitle(f"Sample slices from {class_name} study {study_ids[0]}")
plt.tight_layout()
plt.show()
