# tests/test_visualization.py

import pytest
import numpy as np
import cv2
from pathlib import Path
from utils.visualization import draw_bounding_boxes, overlay_damage_mask

def test_draw_bounding_boxes():
    """Test drawing bounding boxes."""
    # Create a test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        {
            'bbox': (10, 10, 30, 30),
            'confidence': 0.9,
            'class': 1,
            'cropped_image': np.zeros((20, 20, 3), dtype=np.uint8)
        }
    ]
    output_path = Path("tests/data/test_output.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Test drawing boxes
    result = draw_bounding_boxes(test_image, detections, output_path)
    assert result is True
    assert output_path.exists()

    # Cleanup
    output_path.unlink()

def test_overlay_damage_mask():
    """Test overlaying damage mask."""
    # Create test images
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    damage_mask = np.zeros((100, 100), dtype=np.uint8)
    damage_mask[40:60, 40:60] = 1  # Create a damage area

    output_path = Path("tests/data/test_damage_output.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Test overlay
    result = overlay_damage_mask(test_image, damage_mask, output_path)
    assert result is True
    assert output_path.exists()

    # Cleanup
    output_path.unlink()

def test_invalid_inputs():
    """Test visualization with invalid inputs."""
    # Test with None
    assert draw_bounding_boxes(None, [], "output.jpg") is False
    assert overlay_damage_mask(None, None, "output.jpg") is False

    # Test with invalid image
    invalid_image = np.zeros((10, 10))  # 2D array instead of 3D
    assert draw_bounding_boxes(invalid_image, [], "output.jpg") is False
    assert overlay_damage_mask(invalid_image, None, "output.jpg") is False

    # Test with invalid detections
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_detections = [{'invalid_key': 'value'}]
    assert draw_bounding_boxes(test_image, invalid_detections, "output.jpg") is False 