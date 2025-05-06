import cv2
import numpy as np
from collections import deque

def find_path(grid_mask, start_point, end_point):
    height, width = grid_mask.shape
    start_tuple = tuple(start_point)
    end_tuple = tuple(end_point)

    if not (0 <= start_tuple[1] < height and 0 <= start_tuple[0] < width and grid_mask[start_tuple[1], start_tuple[0]] == 255) or \
       not (0 <= end_tuple[1] < height and 0 <= end_tuple[0] < width and grid_mask[end_tuple[1], end_tuple[0]] == 255):
        return None

    queue = deque([(start_tuple, [start_tuple])])
    visited = {start_tuple}

    while queue:
        current_pixel, path = queue.popleft()
        if current_pixel == end_tuple:
            return path

        cx, cy = current_pixel
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Orthogonal moves
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)

            if 0 <= ny < height and 0 <= nx < width and grid_mask[ny, nx] == 255 and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None

def visualize_solution(original_image, solution_path_nodes, graph, path_color=(255, 0, 0)):
    """Draws the solution path on the original image."""
    if original_image is None or not solution_path_nodes:
        return original_image

    solved_image = original_image.copy()
    for i in range(len(solution_path_nodes) - 1):
        start_node = solution_path_nodes[i]
        end_node = solution_path_nodes[i+1]

        # Find the pixel path between these nodes from the graph's edge data
        if isinstance(graph, dict) and (start_node, end_node) in graph:
            path_pixels = graph[(start_node, end_node)].get('path', [])
            for pixel in path_pixels:
                cv2.circle(solved_image, pixel, 1, path_color, -1)  # Draw path pixels

        elif isinstance(graph, dict) and (end_node, start_node) in graph:  # Order might be reversed
            path_pixels = graph[(end_node, start_node)].get('path', [])
            for pixel in path_pixels:
                cv2.circle(solved_image, pixel, 1, path_color, -1)

        cv2.circle(solved_image, start_node, 20, (0, 255, 255), -1)  # Mark start
        cv2.circle(solved_image, end_node, 20, (255, 255, 0), -1)    # Mark end

    return solved_image