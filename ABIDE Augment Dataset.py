

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import re

# Configuration
N_AUG_PER_IMAGE = 11  # Number of transformations per image

# Paths
input_dirs = {'benign': './ABIDE imaging data/benign_raw/', 'malignant': './ABIDE imaging data/malignant_raw/'}
output_dirs = {'benign': './ABIDE imaging data/benign_aug_only/', 'malignant': './ABIDE imaging data/malignant_aug_only/'}

# Create output dirs if they don't exist
for out_dir in output_dirs.values():
    os.makedirs(out_dir, exist_ok=True)

# Extract patient ID from filename using regex
def extract_patient_id(filename):
    match = re.match(r"(patient_\d+)", filename)
    return match.group(1) if match else os.path.splitext(filename)[0]

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment each class
for label in ['benign', 'malignant']:
    input_dir = input_dirs[label]
    output_dir = output_dirs[label]

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Processing class '{label}' with {len(image_files)} original images...")

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        patient_id = extract_patient_id(img_file)

        # Load and prepare image
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate 11 augmentations
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            aug_img = array_to_img(batch[0])
            filename = f"{patient_id}_aug{i}.jpg"
            aug_img.save(os.path.join(output_dir, filename))
            i += 1
            if i >= N_AUG_PER_IMAGE:
                break

    print(f"✅ Finished class '{label}' with {len(image_files) * N_AUG_PER_IMAGE} augmented images.\n")

print("🎉 All image augmentations completed successfully.")
