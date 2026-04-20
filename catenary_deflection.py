import cv2
import numpy as np
import os

print("=== Phase 3: Spatial Deflection & Physics Analysis ===")

# Paths
img_path = "extracted_frames/frame_0050.jpg"
pred_mask_path = "unet_prediction_test.png"

# Load Images
orig_img = cv2.imread(img_path)
mask_img = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

if orig_img is None or mask_img is None:
    print("Error: Could not load the image or prediction mask.")
    exit(1)

# Find components / contours in the AI prediction
contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_cables = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Ignore tiny floating specks under 150 pixels long
    if max(w, h) < 150:
        continue
        
    area = cv2.contourArea(cnt)
    bbox_area = w * float(h)
    
    # Mathematical Extent (contour area / bounding box area)
    # A true 1-pixel thick cable stretching diagonally across a 300x300 bounding box
    # fills barely 1% to 10% of its bounding rectangle.
    # A massive square radiator noise block fills 40% to 100%.
    extent = area / (bbox_area + 1e-6)
    
    if extent < 0.25:
        valid_cables.append(cnt)
        
print(f"Physics Filter: Filtered out noise! Identified {len(valid_cables)} actual unbroken cables.")

for cnt in valid_cables:
    # 1. Extract every pixel associated with the predicted cable
    mask_cable = np.zeros_like(mask_img)
    cv2.drawContours(mask_cable, [cnt], -1, 255, -1)
    y_coords, x_coords = np.where(mask_cable > 0)
    
    if len(x_coords) < 10:
        continue

    # 2. Fit 2nd-Degree Polynomial (Catenary Geometry Approximation)
    # y = Ax^2 + Bx + C
    coeffs = np.polyfit(x_coords, y_coords, 2)
    poly = np.poly1d(coeffs)
    
    x_min, x_max = min(x_coords), max(x_coords)
    
    # 3. Generate smooth continuous curve points for drawing
    x_range = np.linspace(x_min, x_max, 500)
    y_range = poly(x_range)
    
    curve_points = np.int32(np.column_stack((x_range, y_range)))
    # Draw Catenary curve in RED (thickness 3)
    cv2.polylines(orig_img, [curve_points], isClosed=False, color=(0, 0, 255), thickness=3)

    # 4. Calculate Spatial Deflection (Sag)
    # Generate the theoretical perfectly straight taut line endpoints
    y1_str, y2_str = poly(x_min), poly(x_max)
    
    # Draw Straight taut line in BLUE (thickness 2)
    cv2.line(orig_img, (int(x_min), int(y1_str)), (int(x_max), int(y2_str)), (255, 0, 0), 2)
    
    # Mathematical equation of the straight line
    m = (y2_str - y1_str) / (x_max - x_min + 1e-6)
    b = y1_str - m * x_min
    
    # Find max deflection (orthogonal distance)
    max_deflection = 0
    max_x, max_y_curve, max_y_line = 0, 0, 0

    # We test X coordinates along the cable to find where it diverges purely vertically the most
    for test_x in np.linspace(x_min, x_max, 100):
        c_y = poly(test_x)
        l_y = m * test_x + b
        deflection = abs(c_y - l_y)
        
        if deflection > max_deflection:
            max_deflection = deflection
            max_x = test_x
            max_y_curve = c_y
            max_y_line = l_y
            
    # Draw Deflection Measurement (YELLOW arrow pointing from theoretical string to sag)
    cv2.arrowedLine(orig_img, 
                    (int(max_x), int(max_y_line)), 
                    (int(max_x), int(max_y_curve)), 
                    (0, 255, 255), 2, tipLength=0.2)
                    
    # Draw Text
    text = f"Sag: {max_deflection:.1f}px"
    cv2.putText(orig_img, text, (int(max_x) + 15, int(max_y_curve) + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

# Save Final Physics Result
out_path = "phase3_physics_deflection.jpg"
cv2.imwrite(out_path, orig_img)
print(f"Physics Regression Complete! Deflection analysis saved as '{out_path}'.")
