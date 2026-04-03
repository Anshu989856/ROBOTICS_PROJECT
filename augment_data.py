import cv2
import os
import albumentations as A
import numpy as np

# Paths
img_dir = "extracted_frames"
mask_dir = "sam2_masks"
out_img_dir = "augmented_dataset/images"
out_mask_dir = "augmented_dataset/masks"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# Define the Augmentation Pipeline (Fulfilling Proposal Phase 1)
aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # Gaussian Noise
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3), # Elastic Deformations
], additional_targets={'mask': 'mask'})

images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

print("Performing Data Augmentation...")
saved_count = 0

for i, img_name in enumerate(images):
    img = cv2.imread(os.path.join(img_dir, img_name))
    mask_name = img_name.replace(".jpg", "_mask.png")
    mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
    
    if mask is None: 
        continue

    # Save original
    cv2.imwrite(os.path.join(out_img_dir, f"orig_{i}.jpg"), img)
    cv2.imwrite(os.path.join(out_mask_dir, f"orig_{i}.png"), mask)
    saved_count += 1

    # Generate 3 augmented versions per image
    for j in range(3):
        augmented = aug(image=img, mask=mask)
        cv2.imwrite(os.path.join(out_img_dir, f"aug_{i}_{j}.jpg"), augmented['image'])
        cv2.imwrite(os.path.join(out_mask_dir, f"aug_{i}_{j}.png"), augmented['mask'])
        saved_count += 1

print(f"Phase 1 Complete! {saved_count} total samples created in 'augmented_dataset'.")
