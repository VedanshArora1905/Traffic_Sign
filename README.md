# Traffic Sign Damage Detection

A Python-based system for detecting and analyzing damage in traffic signs using computer vision and machine learning.

## Features

- Traffic sign detection using YOLOv5
- Damage classification and severity assessment
- Damage area segmentation
- Automated report generation
- Email notifications with damage reports
- Web interface for uploading and analyzing images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-sign-damage-detection.git
cd traffic-sign-damage-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Unix/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the project root. Use the variables shown in `config/config.yaml` as a template:
  - `YOLO_MODEL_PATH`, `CLASSIFIER_MODEL_PATH`, `SEGMENTER_MODEL_PATH`, `OUTPUT_FOLDER`, `REPORT_FOLDER`, `DETECTION_CONFIDENCE`, `DAMAGE_SEVERITY_LOW`, `DAMAGE_SEVERITY_MEDIUM`, `SENDER_EMAIL`, `SENDER_PASSWORD`, `RECEIVER_EMAIL`, `SMTP_SERVER`, `SMTP_PORT`
- Example:
```
YOLO_MODEL_PATH=./yolov5s.pt
CLASSIFIER_MODEL_PATH=./models/classifier/model.h5
SEGMENTER_MODEL_PATH=./models/segmenter/model.h5
OUTPUT_FOLDER=./data/output
REPORT_FOLDER=./data/reports
DETECTION_CONFIDENCE=0.25
DAMAGE_SEVERITY_LOW=0.2
DAMAGE_SEVERITY_MEDIUM=0.5
SENDER_EMAIL=your@email.com
SENDER_PASSWORD=yourpassword
RECEIVER_EMAIL=receiver@email.com
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
```

## Usage

### Command Line Pipeline

1. Place your traffic sign images in the `data/raw` directory.
2. Run the detection pipeline:
```bash
python main.py
```
3. Check the results in:
   - `data/output/` for processed images
   - `data/reports/` for damage reports

### Web Interface

1. Start the Flask app:
```bash
python app.py
```
2. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
3. Upload an image and view the results and downloadable report.

## Project Structure

```
traffic-sign-damage-detection/
├── app.py
├── main.py
├── train_classifier.py
├── requirements.txt
├── pytest.ini
├── README.md
├── yolov5s.pt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── output/
│   └── reports/
├── models/
│   ├── classifier/
│   ├── detector/
│   └── segmenter/
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   ├── test_visualization.py
│   └── test_emailer.py
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── visualization.py
│   └── emailer.py
├── templates/
│   ├── index.html
│   └── result.html
├── yolov5/  # YOLOv5 codebase (submodule or directory)
└── ...
```

## Dependencies

- Python 3.8+
- opencv-python
- PyYAML
- Pillow
- numpy
- torch
- torchvision
- tensorflow
- ultralytics
- tqdm
- python-dotenv
- pytest, pytest-cov
- black, flake8, mypy

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Configuration

The system can be configured through:
1. Environment variables in `.env` (see above)
2. Configuration file in `config/config.yaml` (uses variables from `.env`)

## Testing

Run tests with coverage:
```bash
pytest
```

## YOLOv5 Directory

The `yolov5/` directory contains the YOLOv5 codebase required for detection. If not present, clone it from [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5).

## License

This project is licensed under the MIT License. If the LICENSE file is not present in the root, refer to the YOLOv5 directory for its license.

## Troubleshooting & Support

- Ensure all required models are present at the paths specified in your `.env` and `config.yaml`.
- If you encounter errors with missing dependencies, re-run `pip install -r requirements.txt`.
- For issues with the web interface, ensure Flask is installed and ports are not blocked.
- For further help, open an issue or discussion on the project repository.

## Acknowledgments

- YOLOv5 for object detection
- OpenCV for image processing
- TensorFlow for damage classification
- PyTorch for segmentation 