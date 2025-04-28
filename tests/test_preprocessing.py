# tests/test_preprocessing.py

import pytest
import numpy as np
import cv2
from pathlib import Path
from utils.preprocessing import preprocess_image

def test_preprocess_image_valid():
    """Test preprocessing with a valid image."""
    # Create a test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_path = Path("tests/data/test_image.jpg")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), test_image)

    # Test preprocessing
    result = preprocess_image(test_path)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape[2] == 3  # Should be a color image

    # Cleanup
    test_path.unlink()

def test_preprocess_image_invalid_path():
    """Test preprocessing with an invalid path."""
    result = preprocess_image("nonexistent.jpg")
    assert result is None

def test_preprocess_image_resize():
    """Test image resizing functionality."""
    # Create a large test image
    test_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    test_path = Path("tests/data/test_large_image.jpg")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), test_image)

    # Test preprocessing with max_dim=500
    result = preprocess_image(test_path, max_dim=500)
    assert result is not None
    assert max(result.shape[:2]) <= 500

    # Cleanup
    test_path.unlink() 