"""
Visualization Utilities

This module contains helper functions for visualizing sign language
recognition results, keypoints, and other data.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logger = logging.getLogger(__name__)


def draw_keypoints_on_image(
    image: np.ndarray,
    keypoints: Dict,
    color_face: Tuple[int, int, int] = (0, 255, 0),
    color_pose: Tuple[int, int, int] = (255, 0, 0),
    color_left_hand: Tuple[int, int, int] = (0, 0, 255),
    color_right_hand: Tuple[int, int, int] = (255, 0, 255),
    point_size: int = 3
) -> np.ndarray:
    """
    Draw extracted keypoints on an image.
    
    Args:
        image: Input image as a numpy array (BGR format)
        keypoints: Dictionary of keypoints from the SignDetector
        color_face: Color for face keypoints (BGR)
        color_pose: Color for pose keypoints (BGR)
        color_left_hand: Color for left hand keypoints (BGR)
        color_right_hand: Color for right hand keypoints (BGR)
        point_size: Size of the keypoint circles
        
    Returns:
        Image with keypoints drawn on it
    """
    if keypoints is None:
        return image
    
    # Make a copy of the image
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    
    # Draw face keypoints
    if 'face' in keypoints and np.any(keypoints['face']):
        for point in keypoints['face']:
            if point[0] > 0 and point[1] > 0:  # Only valid points
                x, y = int(point[0] * w), int(point[1] * h)
                cv2.circle(img_copy, (x, y), point_size, color_face, -1)
    
    # Draw pose keypoints
    if 'pose' in keypoints and np.any(keypoints['pose']):
        for point in keypoints['pose']:
            if point[0] > 0 and point[1] > 0:  # Only valid points
                x, y = int(point[0] * w), int(point[1] * h)
                cv2.circle(img_copy, (x, y), point_size, color_pose, -1)
    
    # Draw left hand keypoints
    if 'left_hand' in keypoints and np.any(keypoints['left_hand']):
        for point in keypoints['left_hand']:
            if point[0] > 0 and point[1] > 0:  # Only valid points
                x, y = int(point[0] * w), int(point[1] * h)
                cv2.circle(img_copy, (x, y), point_size, color_left_hand, -1)
    
    # Draw right hand keypoints
    if 'right_hand' in keypoints and np.any(keypoints['right_hand']):
        for point in keypoints['right_hand']:
            if point[0] > 0 and point[1] > 0:  # Only valid points
                x, y = int(point[0] * w), int(point[1] * h)
                cv2.circle(img_copy, (x, y), point_size, color_right_hand, -1)
    
    return img_copy


def plot_keypoint_history(
    keypoint_history: List[Dict],
    num_frames: int = 10,
    keypoint_index: int = 0,
    hand: str = 'right_hand'
) -> plt.Figure:
    """
    Plot the trajectory of a specific keypoint over time.
    Useful for visualizing hand movements.
    
    Args:
        keypoint_history: List of keypoint dictionaries over time
        num_frames: Number of frames to include in the plot
        keypoint_index: Index of the keypoint to track
        hand: Which hand to track ('right_hand' or 'left_hand')
        
    Returns:
        Matplotlib figure with the plotted trajectory
    """
    if not keypoint_history:
        logger.warning("No keypoint history to plot")
        fig, ax = plt.subplots()
        ax.set_title("No keypoint data available")
        return fig
    
    # Extract x, y coordinates of the specified keypoint
    x_coords = []
    y_coords = []
    
    # Use only the last num_frames frames
    history = keypoint_history[-num_frames:]
    
    for frame_keypoints in history:
        if frame_keypoints and hand in frame_keypoints and np.any(frame_keypoints[hand]):
            if keypoint_index < len(frame_keypoints[hand]):
                point = frame_keypoints[hand][keypoint_index]
                x_coords.append(point[0])
                y_coords.append(point[1])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if x_coords and y_coords:
        # Plot the trajectory
        ax.plot(x_coords, y_coords, 'b-', linewidth=2)
        ax.scatter(x_coords, y_coords, c='r', s=50)
        
        # Add arrows to show direction
        for i in range(len(x_coords) - 1):
            ax.annotate('', 
                xy=(x_coords[i+1], y_coords[i+1]),
                xytext=(x_coords[i], y_coords[i]),
                arrowprops=dict(arrowstyle="->", color='g', lw=1.5)
            )
        
        # Set plot properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.set_title(f"{hand.replace('_', ' ').title()} Keypoint {keypoint_index} Trajectory")
        ax.set_xlabel("X coordinate (normalized)")
        ax.set_ylabel("Y coordinate (normalized)")
        ax.grid(True)
    else:
        ax.set_title(f"No valid keypoints for {hand} index {keypoint_index}")
    
    plt.tight_layout()
    return fig


def add_text_overlay(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Add text overlay to an image.
    
    Args:
        image: Input image
        text: Text to overlay
        position: Position (x, y) to place the text
        font_scale: Size of the font
        color: Text color (BGR)
        thickness: Line thickness of the text
        
    Returns:
        Image with text overlay
    """
    img_copy = image.copy()
    cv2.putText(
        img_copy,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    return img_copy


def create_side_by_side_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    label1: str = "Original",
    label2: str = "Processed"
) -> np.ndarray:
    """
    Create a side-by-side comparison of two images.
    
    Args:
        image1: First image
        image2: Second image
        label1: Label for the first image
        label2: Label for the second image
        
    Returns:
        Combined image with both inputs side by side
    """
    # Ensure both images have the same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Resize the second image if heights don't match
    if h1 != h2:
        scale = h1 / h2
        image2 = cv2.resize(image2, (int(w2 * scale), h1))
    
    # Create the combined image
    combined = np.hstack((image1, image2))
    
    # Add labels
    combined = add_text_overlay(combined, label1, (10, 30))
    combined = add_text_overlay(combined, label2, (w1 + 10, 30))
    
    return combined


# Example usage if this file is run directly
if __name__ == "__main__":
    # Create a sample image
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create sample keypoints
    sample_keypoints = {
        'face': np.array([[0.5, 0.2, 0]]),
        'pose': np.array([[0.5, 0.5, 0]]),
        'left_hand': np.array([[0.3, 0.6, 0]]),
        'right_hand': np.array([[0.7, 0.6, 0]])
    }
    
    # Draw keypoints on the image
    result = draw_keypoints_on_image(sample_image, sample_keypoints)
    
    # Add text overlay
    result = add_text_overlay(result, "Sample Keypoints")
    
    # Display the result
    cv2.imshow('Sample Visualization', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()