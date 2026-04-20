import cv2
import numpy as np
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Master directories 
repo_dir = "CableDrivenRobotCableModel"
out_img_dir = "master_extracted_frames"
out_mask_dir = "master_perfect_masks"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

def process_video(video_path):
    vid_name = os.path.basename(video_path).split('.')[0]
    print(f"[{vid_name}] Initializing Processing...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We want 50 frames evenly spaced from each video
    interval = max(1, total_frames // 50)
    
    # 1. Grab all the target frames to build the background model
    frames_gray = []
    frame_indices = []
    curr = 0
    
    while curr < total_frames and len(frames_gray) < 50:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr)
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
        frame_indices.append(curr)
        curr += interval
        
    if not frames_gray:
        return
        
    # 2. Dynamic Environment Erase (Median Stacking)
    # By taking the statistical median over time, anything that moves (cables, robot) is mathematically deleted.
    # What remains is a perfect image of the empty room specifically for THIS camera angle!
    stacked_frames = np.stack(frames_gray, axis=0) # Shape: (N, H, W)
    empty_room_background = np.median(stacked_frames, axis=0).astype(np.uint8)
    
    saved_count = 0
    
        # 3. Dynamic Foreground Extraction (HYBRID ALGORITHM)
    for i, gray_frame in enumerate(frames_gray):
        
        # --- A. FRANGI RIDGE EXTRACTION ---
        # Frangi extracts all 1-pixel structures perfectly natively (Cables AND Background noise pipes)
        from skimage.filters import frangi
        frangi_img = frangi(gray_frame, sigmas=range(1, 4, 1), black_ridges=True)
        if np.max(frangi_img) > 0:
            frangi_norm = (frangi_img / np.max(frangi_img) * 255).astype(np.uint8)
        else:
            frangi_norm = np.zeros_like(gray_frame)
        _, frangi_binary = cv2.threshold(frangi_norm, 10, 255, cv2.THRESH_BINARY)
        
        # --- B. DYNAMIC ROOM SUBTRACTION ---
        # We subtract the empty room to find general areas of movement
        movement_diff = cv2.absdiff(gray_frame, empty_room_background)
        # Extremely low threshold to catch even the faintest moving shadow
        _, moving_mask = cv2.threshold(movement_diff, 5, 255, cv2.THRESH_BINARY)
        
        # Because faint cables fragment, we use a massive dilation matrix to merge the floating moving 
        # dust into a solid contiguous "Motion Field" bounding the cable trajectory!
        moving_mask = cv2.dilate(moving_mask, np.ones((25, 25), np.uint8), iterations=1)
        
        # --- C. HYBRID INTERSECTION ---
        # The Frangi map has the perfectly high-resolution cables AND static background noise.
        # The Moving Mask ONLY blankets areas with active motion.
        # We intersect them: deleting the static radiators, but preserving the 1-pixel cables!
        hybrid_mask = cv2.bitwise_and(frangi_binary, moving_mask)
        
        # Now we filter the remaining topological noise (e.g. the end-effector box)
        contours, _ = cv2.findContours(hybrid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        perfect_mask = np.zeros_like(hybrid_mask)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Reject camera noise dust
            if max(w, h) < 150:
                continue
                
            area = cv2.contourArea(cnt)
            bbox_area = w * float(h)
            extent = area / (bbox_area + 1e-6)
            
            # The wooden cube end effector is a chunky block (extent > 0.40)
            # The cables are phenomenally thin lines (extent < 0.10)
            if extent < 0.20:
                # One last filter: Must be diagonalish, not a perfectly vertical wall shadow dropping
                [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.degrees(np.arctan2(abs(vy[0]), abs(vx[0])))
                if 10 < angle < 80:
                    cv2.drawContours(perfect_mask, [cnt], -1, 255, -1)
                    
        # Grab Original Color Frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
        ret, color_frame = cap.read()
        
        if ret:
            img_filename = f"{vid_name}_f{i:03d}.jpg"
            mask_filename = f"{vid_name}_f{i:03d}_mask.png"
            
            cv2.imwrite(os.path.join(out_img_dir, img_filename), color_frame)
            cv2.imwrite(os.path.join(out_mask_dir, mask_filename), perfect_mask)
            saved_count += 1
            
    cap.release()
    print(f"[{vid_name}] Successfully isolated {saved_count} perfectly masked frames.")

if __name__ == "__main__":
    start_time = time.time()
    video_files = glob.glob(f"{repo_dir}/**/*.avi", recursive=True)
    if not video_files:
        print("No videos found! Ensure the database is extracted.")
    else:
        print(f"Found {len(video_files)} recorded videos. Launching Neural Parallel Extraction...")
        
        # Fire off all CPU cores to crunch every video simultaneously
        with ProcessPoolExecutor() as executor:
            executor.map(process_video, video_files)
            
        print(f"Mass Extraction Complete in {time.time()-start_time:.1f} seconds! Ground truth saved to 'master_perfect_masks'.")
