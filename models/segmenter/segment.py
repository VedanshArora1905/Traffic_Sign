import tensorflow as tf
import numpy as np
import cv2

def segment_damage(cropped_image, model_path, low_thresh, medium_thresh):
    """
    Segment damage on a cropped sign image and assess severity/type.

    Args:
        cropped_image (np.ndarray): Cropped image of the sign (BGR, uint8).
        model_path (str): Path to the trained segmentation (U-Net) model.
        low_thresh (float): Threshold for low severity.
        medium_thresh (float): Threshold for medium severity.

    Returns:
        mask_resized (np.ndarray): Binary mask of damaged area.
        severity (str): "Low", "Medium", or "High".
        damage_type (str): Type of damage.
    """
    # Load segmentation model (U-Net)
    model = tf.keras.models.load_model(model_path)

    # Preprocess cropped image
    img_resized = cv2.resize(cropped_image, (128, 128))  # U-Net input size
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict damage mask
    pred_mask = model.predict(img_input)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold

    # Resize mask back to original cropped sign size
    mask_resized = cv2.resize(pred_mask, (cropped_image.shape[1], cropped_image.shape[0]))

    # Calculate severity based on damaged area %
    damaged_pixels = np.sum(mask_resized)
    total_pixels = mask_resized.shape[0] * mask_resized.shape[1]
    damage_ratio = damaged_pixels / total_pixels

    if damage_ratio < low_thresh:
        severity = "Low"
    elif damage_ratio < medium_thresh:
        severity = "Medium"
    else:
        severity = "High"

    # Classify type of damage (simple based on color variation)
    damage_type = classify_damage_type(cropped_image, mask_resized)

    return mask_resized, severity, damage_type

def classify_damage_type(cropped_image, mask):
    """
    Simple heuristic: if brightness is very low in damaged area => torn
                      if color faded => color peeling
                      else => scratches/others
    """
    damaged_area = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    hsv = cv2.cvtColor(damaged_area, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])  # V channel = brightness

    if brightness < 50:
        return "Torn Board"
    elif brightness > 150:
        return "Color Peeling"
    else:
        return "Scratches / Faded"