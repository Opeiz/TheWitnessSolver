import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import *

def apply_perspective_transform_with_manual_corners(image):
    """Allows the user to manually select 4 corners of the image and applies a perspective transform."""
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Corners", temp_image)

            if len(points) == 4:
                cv2.destroyWindow("Select Corners")

    points = []
    temp_image = image.copy()
    cv2.imshow("Select Corners", temp_image)
    cv2.setMouseCallback("Select Corners", click_event)
    cv2.waitKey(0)

    if len(points) != 4:
        print("Error: You must select exactly 4 points.")
        return None

    # Define the destination points with padding
    padding = 10
    width = 800
    height = 800
    dst_points = np.array([
        [padding, padding],
        [width - padding - 1, padding],
        [width - padding - 1, height - padding - 1],
        [padding, height - padding - 1]
    ], dtype="float32")

    # Convert points to numpy array
    src_points = np.array(points, dtype="float32")

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transform
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

def quantize_colors_kmeans(image, k_colors):
    """
    Reduces the number of colors in an image using K-Means clustering.

    Args:
        image (numpy.ndarray): The input image (BGR format).
        k_colors (int): The desired number of colors in the output image.

    Returns:
        numpy.ndarray: The image with reduced colors.
    """
    if image is None:
        print("Error: Input image is None.")
        return None

    # Reshape the image to a list of pixels
    pixels = image.reshape((-1, 3)).astype(np.float32)

    # Define criteria for K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Apply K-Means clustering
    _, labels, centers = cv2.kmeans(pixels, k_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit integers
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Create the quantized image
    quantized_image = centers[labels.flatten()].reshape(image.shape)

    return quantized_image

def extract_grid(image, center_x, center_y):
    """Extracts the grid region based on the detected boundaries."""
    grid_rgb_color = image[center_y, center_x]

    # Create the mask based on the HSV range
    lower_bound = grid_rgb_color - np.array([10, 10, 10])
    upper_bound = grid_rgb_color + np.array([10, 10, 10])
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)
    grid_mask = cv2.inRange(image, lower_bound, upper_bound)

    return grid_mask

def detect_start_points_multiple_templates(image, template_paths):
    """
    Detects potential start points in the puzzle image using multiple templates.

    Args:
        image (numpy.ndarray): The input puzzle image (BGR format).
        template_paths (list): A list of paths to template images of the start points.

    Returns:
        list: A list of (x, y) coordinates of the centers of the detected start points.
              Returns an empty list if no start points are found above the threshold.
    """
    if image is None:
        print("Error: Input puzzle image is None.")
        return []

    detected_centers = []

    try:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for template_path in template_paths:
            template = cv2.imread(template_path)
            if template is None:
                print(f"Error: Could not load template image at {template_path}")
                continue

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_height, template_width = template_gray.shape[::-1]

            # Apply template matching
            res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # You can set a threshold to only consider strong matches
            threshold = 0.7  # Adjust this value

            if max_val >= threshold:
                # Get the top-left corner of the matched region
                top_left = max_loc
                bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

                # Calculate the center of the matched region
                center_x = int((top_left[0] + bottom_right[0]) / 2)
                center_y = int((top_left[1] + bottom_right[1]) / 2)

                detected_centers.append((center_x, center_y))

        # Remove duplicate detections if any (e.g., if templates are very similar and overlap)
        unique_centers = list(set(detected_centers))
        print(f"Detected {len(unique_centers)} start points.")

        for center in unique_centers:
            cv2.circle(image, center, 5, (0, 255, 0), -1)

        display_image("Detected Start Points", image)
        return unique_centers

    except Exception as e:
        print(f"An error occurred during template matching: {e}")
        return []

def detect_end_point_manual(image):
    """
    Allows the user to manually click on the end point in the puzzle image.

    Args:
        image (numpy.ndarray): The input puzzle image (BGR format).

    Returns:
        tuple or None: The (x, y) coordinates of the clicked end point,
                       or None if no point is clicked.
    """
    def click_event(event, x, y, flags, param):
        nonlocal end_point
        if event == cv2.EVENT_LBUTTONDOWN:
            end_point = (x, y)
            cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click End Point", temp_image)
            cv2.destroyWindow("Click End Point")

    end_point = None
    temp_image = image.copy()
    cv2.imshow("Click End Point", temp_image)
    cv2.setMouseCallback("Click End Point", click_event)
    cv2.waitKey(0)

    if end_point is None:
        print("Error: No point was clicked.")
    else:
        print(f"End point coordinates: {end_point}")

    return end_point
