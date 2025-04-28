# Traffic Sign Damage Detection

A Python-based system for detecting and analyzing damage in traffic signs using computer vision and machine learning.

## Features

- Traffic sign detection using YOLOv5
- Damage classification and severity assessment
- Damage area segmentation
- Automated report generation
- Email notifications with damage reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-sign-damage-detection.git
cd traffic-sign-damage-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Place your traffic sign images in the `data/raw` directory.

2. Run the detection pipeline:
```bash
python main.py
```

3. Check the results in:
   - `data/output/` for processed images
   - `data/reports/` for damage reports

## Project Structure

```
traffic-sign-damage-detection/
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
│   ├── data/
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
├── main.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## Testing

Run tests with coverage:
```bash
pytest
```

## Configuration

The system can be configured through:
1. Environment variables in `.env`
2. Configuration file in `config/config.yaml`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 for object detection
- OpenCV for image processing
- TensorFlow for damage classification
- PyTorch for segmentation 