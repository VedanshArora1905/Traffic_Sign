import torch
import sys
import os
import cv2

# Add yolov5 directory to system path
sys.path.append(os.path.join(os.getcwd(), "yolov5"))

from yolov5.models.common import DetectMultiBackend

def load_model(weights_path):
    """
    Load the YOLOv5 model from local directory.
    """
    model_path = os.path.join("yolov5", weights_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Load model
    model = torch.load(model_path, map_location=torch.device('cpu'))['model'].float()
    model.eval()  # Set model to evaluation mode
    return model

def detect_signs(image, model_path, confidence_thresh):
    """
    Detect traffic signs in the given image using YOLOv5 model.

    Args:
        image (np.ndarray): Input image (BGR).
        model_path (str): Path to YOLOv5 weights.
        confidence_thresh (float): Detection confidence threshold.

    Returns:
        List[dict]: List of detections with bbox, confidence, and cropped_image.
    """
    model = load_model(model_path)

    # Prepare image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # YOLOv5 default size
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Run detection
    with torch.no_grad():
        pred = model(img)[0]

    # Dummy output format (replace with your bbox parsing)
    detections = [{
        'bbox': (50, 50, 200, 200),  # Dummy bbox (x1, y1, x2, y2)
        'confidence': 0.9,
        'cropped_image': image[50:200, 50:200]  # Crop dummy area
    }]

    # Filter by confidence threshold (for real output)
    # detections = [det for det in detections if det['confidence'] >= confidence_thresh]

    return detections