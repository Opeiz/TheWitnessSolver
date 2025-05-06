import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_image(image_path):
    """Loads an image from the given path."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None
        return img
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None

def display_image(window_name, image):
    """Displays an image in a named window."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


