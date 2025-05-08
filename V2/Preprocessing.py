import cv2
import numpy as np
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grid_extract import *

def save_mask(mask, image_path):
    # Save the mask with the same name in another folder
    output_folder = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Solved_Mask"  # Replace with your desired folder
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, base_name)
    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")

# Load the images
# Unsolved = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Unsolved"  # Replace with your first folder path
Solved = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\Solved"  # Replace with your second folder path

# image_files1 = [f for f in os.listdir(images_folder1) if os.path.isfile(os.path.join(images_folder1, f))]
Solved_list = [f for f in os.listdir(Solved) if os.path.isfile(os.path.join(Solved, f))]

for i in range(len(Solved_list)):

    image_path = os.path.join(Solved, Solved_list[i])
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    img = quantize_colors_kmeans(image, 4)

    cv2.namedWindow("Quantized Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Quantized Image", 800, 800) 
    cv2.moveWindow("Quantized Image", 100, 100)
    cv2.imshow("Quantized Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()