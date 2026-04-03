import cv2
import numpy as np
import os

# Load the original image and the AI's prediction
orig_img = cv2.imread("extracted_frames/frame_0050.jpg")
mask_img = cv2.imread("unet_prediction_test.png", cv2.IMREAD_GRAYSCALE)

# Create a blank image with the same dimensions, colored Neon Green (BGR: 0, 255, 0)
color_mask = np.zeros_like(orig_img)
color_mask[:, :] = (0, 255, 0) 

# Apply the AI's mask to the neon green color
colored_prediction = cv2.bitwise_and(color_mask, color_mask, mask=mask_img)

# Blend the original image and the neon prediction together
alpha = 0.6 # Transparency
overlay_result = cv2.addWeighted(orig_img, 1, colored_prediction, alpha, 0)

# Save the final result
cv2.imwrite("final_report_overlay.jpg", overlay_result)
print("Overlay complete! Saved as 'final_report_overlay.jpg'.")
