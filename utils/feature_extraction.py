# utils/feature_extraction.py

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def extract_hog_features(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract Histogram of Oriented Gradients (HOG) features from the input image.

    Args:
        image (np.ndarray): Input image

    Returns:
        Optional[np.ndarray]: HOG features if successful, None otherwise

    Raises:
        ValueError: If image is invalid
    """
    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")

        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9

        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        
        image_resized = cv2.resize(image, winSize)
        features = hog.compute(image_resized)
        
        return features

    except Exception as e:
        logger.error(f"Failed to extract HOG features: {e}")
        return None

def extract_edge_features(image: np.ndarray) -> Optional[float]:
    """
    Extract edge-based features using Canny edge detector.

    Args:
        image (np.ndarray): Input image

    Returns:
        Optional[float]: Edge density ratio if successful, None otherwise

    Raises:
        ValueError: If image is invalid
    """
    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        
        edge_count = np.sum(edges > 0) / edges.size  # Edge density ratio

        return float(edge_count)

    except Exception as e:
        logger.error(f"Failed to extract edge features: {e}")
        return None

def extract_color_histogram(
    image: np.ndarray,
    bins: Tuple[int, int, int] = (8, 8, 8)
) -> Optional[np.ndarray]:
    """
    Extract a color histogram feature.

    Args:
        image (np.ndarray): Input image
        bins (Tuple[int, int, int]): Number of bins for each channel

    Returns:
        Optional[np.ndarray]: Color histogram if successful, None otherwise

    Raises:
        ValueError: If image is invalid or bins are invalid
    """
    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")
        if not isinstance(bins, tuple) or len(bins) != 3:
            raise ValueError("Invalid bins format")

        hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                           [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    except Exception as e:
        logger.error(f"Failed to extract color histogram: {e}")
        return None
