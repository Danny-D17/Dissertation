import os
import numpy as np
import nibabel as nib
import cv2

# Paths
BASE_DIR = "/Users/daniel/Dissertation/MosMedData Chest CT Scans with COVID-19 Related Findings COVID19_1110 1.0"
STUDIES_DIR = os.path.join(BASE_DIR, "studies")

# Output directory for 2D PNG slices
OUT_DIR = "/Users/daniel/Dissertation/data_slices"
os.makedirs(OUT_DIR, exist_ok=True)

# Classes present in MosMed
CLASS_NAMES = ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]

# How many slices per study to export
SLICES_PER_STUDY = 10      
TARGET_SIZE = (256, 256)   # resize to this for CNN input

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

for class_name in CLASS_NAMES:
    class_in_dir = os.path.join(STUDIES_DIR, class_name)
    class_out_dir = os.path.join(OUT_DIR, class_name)
    os.makedirs(class_out_dir, exist_ok=True)

    if not os.path.isdir(class_in_dir):
        print(f"Skipping {class_name}: folder not found")
        continue

    study_files = sorted([f for f in os.listdir(class_in_dir)
                          if f.lower().endswith(".nii")])
    print(f"{class_name}: {len(study_files)} studies")

    for study_idx, fname in enumerate(study_files):
        study_path = os.path.join(class_in_dir, fname)

        try:
            nii = nib.load(study_path)
            volume = nii.get_fdata()
        except Exception as e:
            print(f"Failed to load {study_path}: {e}")
            continue

        num_slices = volume.shape[-1]
        # Choose SLICES_PER_STUDY indices evenly across the lung
        indices = np.linspace(0, num_slices - 1,
                              num=min(SLICES_PER_STUDY, num_slices),
                              dtype=int)

        for k, idx in enumerate(indices):
            slice_img = volume[:, :, idx]
            slice_img = normalize_to_uint8(slice_img)
            # Resize for CNN
            slice_img = cv2.resize(slice_img, TARGET_SIZE,
                                   interpolation=cv2.INTER_AREA)

            out_name = f"{os.path.splitext(fname)[0]}_slice{idx:03d}.png"
            out_path = os.path.join(class_out_dir, out_name)
            cv2.imwrite(out_path, slice_img)

        if (study_idx + 1) % 50 == 0:
            print(f"{class_name}: processed {study_idx + 1}/{len(study_files)} studies")

print("Done. Slices saved in:", OUT_DIR)
