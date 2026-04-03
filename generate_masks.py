import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 1. Setup paths
input_dir = "extracted_frames"
output_dir = "sam2_masks"
os.makedirs(output_dir, exist_ok=True)

# 2. Load the SAM 2 Model
checkpoint = "./sam2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

print("Loading SAM 2 onto the L40S GPU...")

# L40S optimizations
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Build model and mask generator
sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

# 3. Process the Images
image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')])

for img_path in image_paths:
    print(f"Segmenting {img_path}...")
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate the zero-shot masks
    masks = mask_generator.generate(image_rgb)
    
    if len(masks) == 0:
        continue
        
    # Combine all found segments into one binary mask image for visualization
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for mask_data in masks:
        combined_mask[mask_data['segmentation']] = 255
        
    # Save the output
    filename = os.path.basename(img_path)
    out_path = os.path.join(output_dir, filename.replace('.jpg', '_mask.png'))
    cv2.imwrite(out_path, combined_mask)

print(f"\nPhase 1 Complete! Masks saved to the '{output_dir}' folder.")
