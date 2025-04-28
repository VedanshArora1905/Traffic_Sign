#!/usr/bin/env python3
"""
Road Sign Damage Detection Pipeline
"""

import os
import yaml
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import torch
import tensorflow as tf
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {
    'yolo': None,
    'classifier': None,
    'segmenter': None
}

def get_cached_model(model_type: str, model_path: str):
    """Get a cached model or load it if not cached"""
    if _model_cache[model_type] is None:
        if model_type == 'yolo':
            _model_cache[model_type] = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            _model_cache[model_type] = tf.keras.models.load_model(model_path)
    return _model_cache[model_type]

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess an image for sign detection.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Convert to RGB (from BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large (maintain aspect ratio)
        max_dimension = 1280
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        return image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise PipelineError("Image preprocessing error")

def detect_signs(
    image: np.ndarray,
    model_path: str,
    confidence_thresh: float
) -> List[Dict]:
    """
    Detect traffic signs in an image using YOLO model.
    
    Args:
        image (np.ndarray): Preprocessed input image
        model_path (str): Path to YOLO model
        confidence_thresh (float): Detection confidence threshold
        
    Returns:
        List[Dict]: List of detected signs with their bounding boxes and cropped images
    """
    try:
        # Get cached model
        model = get_cached_model('yolo', model_path)
        model.conf = confidence_thresh
        
        # Run inference
        results = model(image)
        
        # Process detections
        detected_signs = []
        for det in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            
            # Convert to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Crop the sign
            cropped_sign = image[y1:y2, x1:x2]
            
            detected_signs.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class': int(cls),
                'cropped_image': cropped_sign
            })
            
        return detected_signs
        
    except Exception as e:
        logger.error(f"Sign detection failed: {str(e)}")
        raise PipelineError("Sign detection error")

def classify_sign(
    cropped_sign: np.ndarray,
    model_path: str,
    damage_thresh: float
) -> Tuple[bool, str, float]:
    """
    Classify a traffic sign as damaged or intact.
    
    Args:
        cropped_sign (np.ndarray): Cropped image of the sign
        model_path (str): Path to the classifier model
        damage_thresh (float): Threshold for damage classification
        
    Returns:
        Tuple[bool, str, float]: (is_damaged, original_label, damage_probability)
    """
    try:
        # Get cached model
        model = get_cached_model('classifier', model_path)
        
        # Preprocess image for classification
        img = cv2.resize(cropped_sign, (224, 224))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        
        # Get predictions
        predictions = model.predict(img)
        damage_prob = float(predictions[0][0])
        
        # Get original label (assuming model has multiple outputs)
        original_label = "Unknown"  # This should be replaced with actual label prediction
        
        # Determine if damaged
        is_damaged = damage_prob > damage_thresh
        
        return is_damaged, original_label, damage_prob
        
    except Exception as e:
        logger.error(f"Sign classification failed: {str(e)}")
        raise PipelineError("Sign classification error")

def segment_damage(
    cropped_sign: np.ndarray,
    model_path: str,
    low_thresh: float,
    medium_thresh: float
) -> Tuple[np.ndarray, str, str]:
    """
    Segment damaged areas in a traffic sign.
    
    Args:
        cropped_sign (np.ndarray): Cropped image of the sign
        model_path (str): Path to the segmentation model
        low_thresh (float): Threshold for low severity damage
        medium_thresh (float): Threshold for medium severity damage
        
    Returns:
        Tuple[np.ndarray, str, str]: (damage_mask, severity, damage_type)
    """
    try:
        # Get cached model
        model = get_cached_model('segmenter', model_path)
        
        # Preprocess image for segmentation
        img = cv2.resize(cropped_sign, (256, 256))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        
        # Get segmentation mask
        mask = model.predict(img)[0]
        mask = (mask > 0.5).astype(np.uint8)  # Binary threshold
        
        # Calculate damage severity
        damage_ratio = np.mean(mask)
        
        # Determine severity
        if damage_ratio < low_thresh:
            severity = "Low"
        elif damage_ratio < medium_thresh:
            severity = "Medium"
        else:
            severity = "High"
            
        # Determine damage type (simplified)
        damage_type = "Unknown"  # This should be replaced with actual damage type classification
        
        return mask, severity, damage_type
        
    except Exception as e:
        logger.error(f"Damage segmentation failed: {str(e)}")
        raise PipelineError("Damage segmentation error")

def generate_report(
    bbox: Tuple[int, int, int, int],
    label: str,
    severity: str,
    damage_type: str,
    output_dir: str
) -> str:
    """
    Generate a report for a damaged sign.
    
    Args:
        bbox (Tuple[int, int, int, int]): Bounding box coordinates
        label (str): Sign label
        severity (str): Damage severity
        damage_type (str): Type of damage
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the generated report
    """
    try:
        # Create report directory if it doesn't exist
        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"damage_report_{timestamp}.txt"
        
        # Generate report content
        report_content = f"""
Damage Report
-------------
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Sign Label: {label}
Damage Severity: {severity}
Damage Type: {damage_type}
Location: {bbox}
        """
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Generated report: {report_path}")
        return str(report_path)
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise PipelineError("Report generation error")

def send_email(
    report_path: str,
    sender_email: str,
    sender_password: str,
    receiver_email: str,
    smtp_server: str,
    smtp_port: int
) -> None:
    """
    Send an email with the damage report.
    
    Args:
        report_path (str): Path to the report file
        sender_email (str): Sender's email address
        sender_password (str): Sender's email password
        receiver_email (str): Receiver's email address
        smtp_server (str): SMTP server address
        smtp_port (int): SMTP server port
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "Traffic Sign Damage Report"
        
        # Add body
        body = "Please find attached the damage report for a traffic sign."
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach report
        with open(report_path, 'rb') as f:
            report = MIMEApplication(f.read(), _subtype='txt')
            report.add_header('Content-Disposition', 'attachment', filename=Path(report_path).name)
            msg.attach(report)
            
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            
        logger.info(f"Email sent successfully to {receiver_email}")
        
    except Exception as e:
        logger.error(f"Email sending failed: {str(e)}")
        raise PipelineError("Email sending error")

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

class PipelineError(Exception):
    """Custom exception for pipeline processing errors"""
    pass

def load_and_validate_config(config_path: str) -> Dict:
    """
    Load and validate configuration file with strict checking
    Returns validated config dictionary
    Raises ConfigError on validation failure
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Validate structure
        required_structure = {
            'paths': {
                'yolo_model': str,
                'classifier_model': str,
                'segmenter_model': str,
                'output_folder': str,
                'report_folder': str
            },
            'thresholds': {
                'detection_confidence': (float, int),
                'damage_severity_low': (float, int),
                'damage_severity_medium': (float, int)
            },
            'email': {
                'sender_email': str,
                'sender_password': str,
                'receiver_email': str,
                'smtp_server': str,
                'smtp_port': int
            }
        }

        for section, fields in required_structure.items():
            if section not in config:
                raise ConfigError(f"Missing section: {section}")
            for field, field_type in fields.items():
                if field not in config[section]:
                    raise ConfigError(f"Missing field: {section}.{field}")
                if not isinstance(config[section][field], field_type):
                    raise ConfigError(f"Invalid type for {section}.{field}")

        # Convert paths to absolute and create directories
        for folder_type in ['output_folder', 'report_folder']:
            path = Path(config['paths'][folder_type])
            path.mkdir(parents=True, exist_ok=True)
            config['paths'][folder_type] = str(path.resolve())

        return config

    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML: {str(e)}")
    except Exception as e:
        raise ConfigError(f"Configuration error: {str(e)}")

def validate_image_format(image_path: str) -> bool:
    """
    Validate image format and size.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        if not Path(image_path).suffix.lower() in valid_extensions:
            logger.warning(f"Invalid file extension: {image_path}")
            return False
            
        # Check file size
        file_size = Path(image_path).stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            logger.warning(f"File too large: {image_path} ({file_size/1024/1024:.1f}MB)")
            return False
            
        # Check image dimensions
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False
            
        height, width = img.shape[:2]
        if width < 100 or height < 100:
            logger.warning(f"Image too small: {image_path} ({width}x{height})")
            return False
            
        if width > 4000 or height > 4000:
            logger.warning(f"Image too large: {image_path} ({width}x{height})")
            return False
            
        return True
        
    except Exception as e:
        logger.warning(f"Image validation failed for {image_path}: {str(e)}")
        return False

def get_image_paths(input_dir: str) -> List[str]:
    """Get list of valid image paths from directory"""
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Get all image files
        image_paths = [
            str(p) for p in input_path.iterdir() 
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ]

        if not image_paths:
            logger.warning(f"No valid images found in {input_dir}")
            return []
            
        # Validate each image
        valid_paths = []
        for img_path in image_paths:
            if validate_image_format(img_path):
                valid_paths.append(img_path)
            else:
                logger.warning(f"Skipping invalid image: {img_path}")
                
        return valid_paths

    except Exception as e:
        logger.error(f"Failed to get image paths: {str(e)}")
        raise PipelineError("Image path collection failed")

def process_single_sign(
    cropped_sign: "np.ndarray",
    sign_bbox: Tuple,
    config: Dict,
    image_path: str,
    sign_idx: int
) -> Optional[str]:
    """Process a single detected sign through classification and damage analysis"""
    try:
        # Classify sign
        is_damaged, original_label, damage_prob = classify_sign(
            cropped_sign,
            model_path=config['paths']['classifier_model'],
            damage_thresh=config['thresholds']['damage_severity_low']
        )

        if not is_damaged:
            logger.info(f"[{sign_idx}] Sign intact in {image_path}")
            return None

        # Segment damage if classified as damaged
        damage_mask, severity, damage_type = segment_damage(
            cropped_sign,
            model_path=config['paths']['segmenter_model'],
            low_thresh=config['thresholds']['damage_severity_low'],
            medium_thresh=config['thresholds']['damage_severity_medium']
        )

        # Generate report
        report_path = generate_report(
            bbox=sign_bbox,
            label=original_label,
            severity=severity,
            damage_type=damage_type,
            output_dir=config['paths']['report_folder']
        )

        # Send email notification
        send_email(
            report_path=report_path,
            sender_email=config['email']['sender_email'],
            sender_password=config['email']['sender_password'],
            receiver_email=config['email']['receiver_email'],
            smtp_server=config['email']['smtp_server'],
            smtp_port=config['email']['smtp_port']
        )

        return report_path

    except Exception as e:
        logger.error(f"Sign processing failed: {str(e)}")
        raise PipelineError("Sign processing error")

def process_image(image_path: str, config: Dict) -> bool:
    """Full processing pipeline for a single image"""
    try:
        if not validate_image_format(image_path):
            return False

        logger.info(f"Processing image: {image_path}")
        image = preprocess_image(image_path)

        # Detect signs in image
        detected_signs = detect_signs(
            image=image,
            model_path=config['paths']['yolo_model'],
            confidence_thresh=config['thresholds']['detection_confidence']
        )

        if not detected_signs:
            logger.info(f"No signs detected in {image_path}")
            return True

        # Process each detected sign
        for idx, sign in enumerate(detected_signs, 1):
            try:
                process_single_sign(
                    cropped_sign=sign['cropped_image'],
                    sign_bbox=sign['bbox'],
                    config=config,
                    image_path=image_path,
                    sign_idx=idx
                )
            except PipelineError:
                continue  # Error already logged

        return True

    except Exception as e:
        logger.error(f"Image processing failed for {image_path}: {str(e)}")
        return False

def validate_model(model_path: str, model_type: str) -> bool:
    """
    Validate that a model file exists and can be loaded.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model ('yolo', 'classifier', or 'segmenter')
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return False
            
        # Try loading the model
        if model_type == 'yolo':
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            model = tf.keras.models.load_model(model_path)
            
        # Basic validation
        if model is None:
            logger.error(f"Failed to load model: {model_path}")
            return False
            
        logger.info(f"Successfully validated {model_type} model: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed for {model_path}: {str(e)}")
        return False

def validate_models(config: Dict) -> bool:
    """
    Validate all required models.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        bool: True if all models are valid, False otherwise
    """
    models_to_validate = {
        'yolo': config['paths']['yolo_model'],
        'classifier': config['paths']['classifier_model'],
        'segmenter': config['paths']['segmenter_model']
    }
    
    for model_type, model_path in models_to_validate.items():
        if not validate_model(model_path, model_type):
            return False
            
    return True

def main():
    """Main execution function"""
    try:
        # Load configuration
        config = load_and_validate_config('config/config.yaml')
        logger.info("Configuration validated successfully")
        
        # Validate models
        if not validate_models(config):
            logger.critical("Model validation failed")
            return

        # Get images to process
        image_paths = get_image_paths("data/raw/")
        if not image_paths:
            logger.warning("No images to process")
            return

        logger.info(f"Starting processing of {len(image_paths)} images")

        # Process images in parallel
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        def process_with_progress(img_path):
            try:
                return process_image(img_path, config)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                return False

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create progress bar
            with tqdm(total=len(image_paths), desc="Processing images") as pbar:
                # Submit all tasks
                future_to_path = {
                    executor.submit(process_with_progress, img_path): img_path 
                    for img_path in image_paths
                }
                
                # Process results as they complete
                success_count = 0
                for future in concurrent.futures.as_completed(future_to_path):
                    img_path = future_to_path[future]
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {str(e)}")
                    finally:
                        pbar.update(1)

        logger.info(
            f"Processing complete. Success: {success_count}/{len(image_paths)}"
        )

    except ConfigError as e:
        logger.critical(f"Configuration error: {str(e)}")
    except PipelineError as e:
        logger.critical(f"Processing error: {str(e)}")
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()