import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from utils import *

def apply_perspective_transform_with_manual_corners(image):
    """Allows the user to manually select 4 corners of the image and applies a perspective transform."""

    windows_name = "Select Corners"

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(windows_name, temp_image)

            if len(points) == 4:
                cv2.destroyWindow(windows_name)

    points = []
    temp_image = image.copy()
    cv2.namedWindow(windows_name)
    cv2.moveWindow(windows_name, 100, 100)
    cv2.imshow(windows_name, temp_image)
    cv2.setMouseCallback(windows_name, click_event)
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

def detect_start_point_manual(image):

    windows_name = "Click Start Point"

    def click_event(event, x, y, flags, param):
        nonlocal end_point
        if event == cv2.EVENT_LBUTTONDOWN:
            end_point = (x, y)
            cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(windows_name, temp_image)
            cv2.destroyWindow(windows_name)

    end_point = None
    temp_image = image.copy()
    cv2.namedWindow(windows_name)
    cv2.moveWindow(windows_name, 100, 100)
    cv2.imshow(windows_name, temp_image)
    cv2.setMouseCallback(windows_name, click_event)
    cv2.waitKey(0)

    if end_point is None:
        print("Error: No point was clicked.")
    else:
        print(f"End point coordinates: {end_point}")

    return end_point

def detect_end_point_manual(image):

    windows_name = "Click End Point"

    def click_event(event, x, y, flags, param):
        nonlocal end_point
        if event == cv2.EVENT_LBUTTONDOWN:
            end_point = (x, y)
            cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(windows_name, temp_image)
            cv2.destroyWindow(windows_name)

    end_point = None
    temp_image = image.copy()
    cv2.namedWindow(windows_name)
    cv2.moveWindow(windows_name, 100, 100)
    cv2.imshow(windows_name, temp_image)
    cv2.setMouseCallback(windows_name, click_event)
    cv2.waitKey(0)

    if end_point is None:
        print("Error: No point was clicked.")
    else:
        print(f"End point coordinates: {end_point}")

    return end_point
