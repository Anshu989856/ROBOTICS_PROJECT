import torch
import cv2
import os
import numpy as np
from unet_model import UNet

print("Loading Trained U-Net Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize model and load your trained weights
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("cable_unet_midterm.pth"))
model.eval() # Set to evaluation mode

# 2. Grab a testing image from a random dynamic background
# Testing against a completely different camera perspective
img_path = "master_extracted_frames/baumer_video0048_f020.jpg"
image = cv2.imread(img_path)
orig_h, orig_w = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Preprocess for the U-Net
img_resized = cv2.resize(image_rgb, (256, 256))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img_tensor = img_tensor.to(device)

print("Running pure generalized U-Net prediction...")
# 4. Predict!
with torch.no_grad():
    pred_mask = model(img_tensor)

# 5. Post-process the output
pred_mask = pred_mask.squeeze().cpu().numpy()
pred_mask = (pred_mask > 0.1).astype(np.uint8) * 255 # Lower threshold to catch faint learning
pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h)) # Scale back to original resolution

# --- ALL MANUAL STATIC BLACKOUT BOXES HAVE BEEN PERMANENTLY REMOVED ---

# 6. Save the comparison
out_path = "unet_prediction_test.png"
cv2.imwrite(out_path, pred_mask_resized)
print(f"Phase 2 Complete! AI prediction saved as '{out_path}'.")

# --- DIAGNOSTIC GROUND TRUTH CHECK ---
# Let's verify if the automated extraction pipeline actually created a valid mask here!
mask_path = img_path.replace("master_extracted_frames", "master_perfect_masks").replace(".jpg", "_mask.png")
if os.path.exists(mask_path):
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if cv2.countNonZero(gt) == 0:
        print("\nWARNING: Ground Truth Mask is PURE BLACK! The extraction failed to see cables.")
    else:
        print("\nSUCCESS: Ground Truth Mask contains valid pixels.")
else:
    print(f"\nWARNING: Could not find ground truth mask at {mask_path}")
