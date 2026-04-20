import cv2
import os
import albumentations as A
import numpy as np

# Paths
img_dir = "master_extracted_frames"
mask_dir = "master_perfect_masks"
out_img_dir = "master_augmented_dataset/images"
out_mask_dir = "master_augmented_dataset/masks"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# Define the Augmentation Pipeline (Fulfilling Proposal Phase 1)
# Removed deprecated `var_limit` and `alpha_affine` parameters; leveraging native core defaults.
aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.5), # Gaussian Noise
    A.ElasticTransform(alpha=1, sigma=50, p=0.3), # Elastic Deformations
], additional_targets={'mask': 'mask'})

images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

print(f"Processing {len(images)} base frames into 5,200 augmentations! This will take a few minutes...")
saved_count = 0

for i, img_name in enumerate(images):
    img = cv2.imread(os.path.join(img_dir, img_name))
    mask_name = img_name.replace(".jpg", "_mask.png")
    mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
    
    if mask is None: 
        continue

    # Save original (Zero-padded to handle thousands securely)
    cv2.imwrite(os.path.join(out_img_dir, f"orig_{i:04d}.jpg"), img)
    cv2.imwrite(os.path.join(out_mask_dir, f"orig_{i:04d}.png"), mask)
    saved_count += 1

    # Generate 3 augmented versions per image
    for j in range(3):
        augmented = aug(image=img, mask=mask)
        cv2.imwrite(os.path.join(out_img_dir, f"aug_{i:04d}_{j}.jpg"), augmented['image'])
        cv2.imwrite(os.path.join(out_mask_dir, f"aug_{i:04d}_{j}.png"), augmented['mask'])
        saved_count += 1
        
    if (i + 1) % 100 == 0:
        print(f"[{i + 1}/{len(images)}] Generated {saved_count} augmented shards...")

print(f"Phase 1 Complete! {saved_count} total samples created in 'master_augmented_dataset'.")
