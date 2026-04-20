"""
Minimal g_diffuser_bot.py - Just the expand_image function needed by base.py
"""
import cv2
import numpy as np


def expand_image(
    image,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    left: int = 0,
    softness: float = 0.0,
):
    """
    Expand image by adding borders
    
    Args:
        image: Input image
        top, right, bottom, left: Border sizes
        softness: Edge softness (not used in basic version)
    
    Returns:
        Expanded image and mask
    """
    if top == 0 and right == 0 and bottom == 0 and left == 0:
        return image, None
    
    # Get original dimensions
    origin_h, origin_w = image.shape[:2]
    
    # Calculate new dimensions
    new_h = origin_h + top + bottom
    new_w = origin_w + left + right
    
    # Create new image with borders
    new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # Copy original image to center
    new_image[top:top+origin_h, left:left+origin_w] = image
    
    # Create mask (white for original area, black for expanded area)
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    mask[top:top+origin_h, left:left+origin_w] = 255
    
    # Invert mask (black for original, white for expanded)
    mask = 255 - mask
    
    return new_image, mask
