# utils/visualization.py

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Union[Tuple[int, int, int, int], float, int, np.ndarray]]],
    output_path: Union[str, Path]
) -> bool:
    """
    Draws bounding boxes around detected traffic signs.

    Args:
        image (np.ndarray): Input image
        detections (List[Dict]): List of detections with bounding boxes
        output_path (Union[str, Path]): Path to save the output image

    Returns:
        bool: True if successful, False otherwise

    Raises:
        ValueError: If image is invalid or detections are malformed
        IOError: If output path is invalid
    """
    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")

        # Create a copy of the image to avoid modifying the original
        output_image = image.copy()

        for det in detections:
            try:
                bbox = det['bbox']
                if not isinstance(bbox, tuple) or len(bbox) != 4:
                    raise ValueError(f"Invalid bounding box format: {bbox}")
                
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(output_image, "Traffic Sign", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            except KeyError as e:
                logger.warning(f"Missing key in detection: {e}")
                continue

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the image
        cv2.imwrite(str(output_path), output_image)
        logger.info(f"Bounding boxes drawn and saved at {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to draw bounding boxes: {e}")
        return False

def overlay_damage_mask(
    cropped_sign: np.ndarray,
    damage_mask: np.ndarray,
    output_path: Union[str, Path]
) -> bool:
    """
    Overlays the damage mask on the traffic sign image and saves it.

    Args:
        cropped_sign (np.ndarray): Cropped traffic sign image
        damage_mask (np.ndarray): Binary mask indicating damaged areas
        output_path (Union[str, Path]): Path to save the output image

    Returns:
        bool: True if successful, False otherwise

    Raises:
        ValueError: If input images are invalid
        IOError: If output path is invalid
    """
    try:
        if cropped_sign is None or not isinstance(cropped_sign, np.ndarray):
            raise ValueError("Invalid cropped sign image")
        if damage_mask is None or not isinstance(damage_mask, np.ndarray):
            raise ValueError("Invalid damage mask")

        # Create a red mask
        red_mask = np.zeros_like(cropped_sign)
        red_mask[:, :, 2] = 255  # Red channel

        # Blend the red mask where damage is detected
        mask_3ch = np.stack([damage_mask]*3, axis=-1)
        damaged_area = np.where(mask_3ch == 1, red_mask, 0)
        blended = cv2.addWeighted(cropped_sign, 1.0, damaged_area, 0.5, 0)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the image
        cv2.imwrite(str(output_path), blended)
        logger.info(f"Damage mask overlay saved at {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to overlay damage mask: {e}")
        return False
