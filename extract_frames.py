import cv2
import os
import glob

repo_dir = "CableDrivenRobotCableModel"
output_dir = "extracted_frames"
os.makedirs(output_dir, exist_ok=True)

video_files = glob.glob(f"{repo_dir}/**/*.avi", recursive=True)

if not video_files:
    print("No .avi files found!")
else:
    # Let's use a different video this time for more variety
    video_path = video_files[3] 
    print(f"Extracting 100 images from: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate interval to get exactly 100 frames
    interval = max(1, total_frames // 100)
    
    saved_count = 0
    frame_idx = 0

    while saved_count < 100:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        
        saved_count += 1
        frame_idx += interval

    cap.release()
    print(f"Done! Successfully extracted {saved_count} frames into '{output_dir}'.")
