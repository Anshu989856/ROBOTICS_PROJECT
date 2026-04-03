import cv2
import numpy as np
import os
import shutil
import torch
from sam2.build_sam import build_sam2_video_predictor

print("--- Initializing Industrial Video Tracking Pipeline ---")

# --- NEW STEP: Prepare a SAM2-Strict Directory ---
original_dir = "../extracted_frames"
sam2_video_dir = "../extracted_frames_sam2"
os.makedirs(sam2_video_dir, exist_ok=True)

print("Prepping frames for strict SAM 2 naming conventions...")
for f in os.listdir(original_dir):
    if f.startswith("frame_") and f.endswith(".jpg"):
        # Convert 'frame_0025.jpg' into '0025.jpg'
        strict_name = f.replace("frame_", "")
        shutil.copy2(os.path.join(original_dir, f), os.path.join(sam2_video_dir, strict_name))

# 1. Setup Video Predictor
checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 2. ArUco Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

output_dir = "video_tracked_masks"
os.makedirs(output_dir, exist_ok=True)

# 3. 'Seed' the First Frame (using the new pure-integer name)
first_frame_path = os.path.join(sam2_video_dir, "0000.jpg")
first_frame = cv2.imread(first_frame_path)

if first_frame is None:
    print(f"Error loading frame at {first_frame_path}")
    exit()

corners, ids, _ = detector.detectMarkers(first_frame)
inference_state = predictor.init_state(video_path=sam2_video_dir)

if ids is not None:
    print("Found ArUco anchor in Frame 0. Seeding the temporal tracker...")
    for i in range(len(ids)):
        c = corners[i][0]
        center = np.mean(c, axis=0)
        
        # Radial Offsets to hit the cable
        for corner in c:
            vector = corner - center
            outer_point = center + vector * 1.5
            
            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i,
                points=np.array([outer_point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

# 4. Propagate the tracking through all 100 frames
print("Propagating tracking across all frames (Temporal Memory)...")
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    combined_mask = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
    for i, out_obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, mask[0])
    
    # Save using your standard naming convention so your U-Net script can read it
    cv2.imwrite(f"{output_dir}/frame_{out_frame_idx:04d}_mask.png", combined_mask)

print("\nTracking Complete. Temporal masks saved in 'video_tracked_masks'.")
