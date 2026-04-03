import cv2
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Model Setup
checkpoint = "checkpoints/sam2_hiera_large.pt" 
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 2. ArUco Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

input_dir = "../extracted_frames"
output_dir = "perfect_masks"
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')])

for img_path in image_paths:
    image = cv2.imread(img_path)
    if image is None: continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    corners, ids, rejected = detector.detectMarkers(image)
    final_cable_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    if ids is not None:
        print(f"Isolating cables in {os.path.basename(img_path)}...")
        for i in range(len(ids)):
            c = corners[i][0]
            center = np.mean(c, axis=0)
            
            # --- TASK 1: MASK THE CUBE ---
            # Use center point to find the block
            m_cube, _, _ = predictor.predict(point_coords=np.array([center]), point_labels=np.array([1]), multimask_output=False)
            
            # --- TASK 2: MASK CUBE + CABLE ---
            # Use radial offsets (1.7x extension) to grab the cable
            outer_points = [center + (corner - center) * 1.7 for corner in c]
            m_combined, _, _ = predictor.predict(point_coords=np.array(outer_points), point_labels=np.array([1,1,1,1]), multimask_output=False)
            
            # --- TASK 3: SUBTRACTION ---
            # Cable = Combined - Cube
            cable_only = cv2.subtract(m_combined[0].astype(np.uint8)*255, m_cube[0].astype(np.uint8)*255)
            
            # --- TASK 4: CLEANUP ---
            # Remove small noise and "skeletonize" to get a thin line
            kernel = np.ones((3,3), np.uint8)
            cable_only = cv2.morphologyEx(cable_only, cv2.MORPH_OPEN, kernel)
            final_cable_mask = cv2.bitwise_or(final_cable_mask, cable_only)
    
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '_mask.png')), final_cable_mask)

print("\nSubtraction complete. These masks should be significantly thinner.")
