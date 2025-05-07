import cv2
import numpy as np
import os
import random

from utils import *
from findpath import *
from grid_extract import *

if __name__ == "__main__":
    print("Welcome to The Witness Puzzle Solver!")

    # Option to load an image from a file or capture from live feed
    # input_method = input("Enter 'file' to load from a file or 'live' to use live feed: ").lower()
    input_method = "file"

    if input_method == 'file':
        # image_path = input("Enter the path to the puzzle image: ")
        img_folder = "C:\\Users\\jaopa\\Documents\\TheWitnessSolver\\img"
        image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_path = os.path.join(img_folder, random.choice(image_files))
        puzzle_image = load_image(image_path)
        if puzzle_image is None:
            exit()
    # elif input_method == 'live':
    #     puzzle_image = capture_live_feed()
    #     if puzzle_image is None:
    #         exit()
    # else:
    #     print("Invalid input method.")
    #     exit()

    # Basic image processing
    # processed_image = preprocess_image(puzzle_image)
    processed_image = puzzle_image.copy()
    # display_image("Processed Image", processed_image)

    # Apply perspective transform
    processed_image = apply_perspective_transform_with_manual_corners(processed_image)
    # display_image("Transformed Image", processed_image)

    processed_image = quantize_colors_kmeans(processed_image, 2)
    # display_image("Quantized Image", processed_image)
    
    start_point = detect_start_point_manual(processed_image)
    end_point = detect_end_point_manual(processed_image)
    # display_image("Start Point", processed_image)

    # Extract the grid
    grid = extract_grid(processed_image, start_point[0], start_point[1])
    path = find_path(grid, start_point, end_point)

    solution = visualize_solution(processed_image, path, grid)
    
    display_image("Solved Puzzle", solution)

    print("Exiting.")