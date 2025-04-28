# utils/preprocessing.py

import cv2
import numpy as np
import logging
from typing import Union, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def preprocess_image(
    image_path: Union[str, Path],
    max_dim: int = 800
) -> Optional[np.ndarray]:
    """
    Loads an image from disk, resizes if needed, and returns the OpenCV image.

    Args:
        image_path (Union[str, Path]): Path to the input image
        max_dim (int): Maximum dimension for resizing

    Returns:
        Optional[np.ndarray]: Preprocessed image if successful, None otherwise

    Raises:
        ValueError: If image path is invalid or image cannot be loaded
        IOError: If file cannot be read
    """
    try:
        # Convert to Path object
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Invalid image path: {image_path}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Optional resizing for large images
        height, width = image.shape[:2]
        
        if max(height, width) > max_dim:
            scaling_factor = max_dim / float(max(height, width))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        logger.info(f"Successfully preprocessed image: {image_path}")
        return image

    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        return None
