import tensorflow as tf
import numpy as np
import cv2

def classify_sign(cropped_image, model_path, damage_thresh):
    """
    Classify a cropped road sign image for damage and original sign class.

    Args:
        cropped_image (np.ndarray): Cropped image of the sign (BGR, uint8).
        model_path (str): Path to the trained classifier model.
        damage_thresh (float): Threshold for classifying as damaged.

    Returns:
        is_damaged (bool): True if sign is damaged, else False.
        original_label (str): Predicted label for the sign class.
        original_sign_prob (float): Probability of the predicted sign class.
    """
    # Load classifier model
    model = tf.keras.models.load_model(model_path)

    # Preprocess cropped image
    img_resized = cv2.resize(cropped_image, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_input)
    
    # Model output: [damage_status_probability, original_sign_classes_probabilities...]
    damage_prob = predictions[0][0]   # probability that it is damaged
    original_sign_probs = predictions[0][1:]  # probabilities of sign classes

    is_damaged = damage_prob >= damage_thresh

    # Predict original sign class (highest probability)
    original_sign_idx = np.argmax(original_sign_probs)
    original_sign_prob = original_sign_probs[original_sign_idx]

    # Label signs as numbers (0, 1, 2, ..., n)
    original_label = f"Sign_{original_sign_idx}"

    return is_damaged, original_label, original_sign_prob