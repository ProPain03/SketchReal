
# SketchReal - HTML Code from UI image

A smart dual-model system that identifies UI elements in both hand-drawn sketches and real UI screenshots, providing accurate bounding box detection and classification. Then converting it to HTML code.

## 🎯 Project Overview

SketchReal Phase 1 combines two powerful approaches:
- **Unified CNN Model**: Trained on QuickDraw sketches and synthetic UI data for sketch element detection
- **ScreenRecognition Model**: Pre-trained model for detecting UI elements in real screenshots
- **Smart Fusion Engine**: Intelligently combines results based on input type detection

## 🏗️ Project Structure

```
SketchReal/
├── app/                          # Streamlit web application
│   ├── app.py
│   ├── requirements.txt
│   ├── .streamlit/
│   │   └── secrets.toml
│   └── utils/
│       ├── backend1.py
│       ├── style.css
│       └── ui_helpers.py
├── config/                       # Configuration files
│   ├── __init__.py
│   └── config.py                # Main configuration settings
├── src/                         # Core source code
│   ├── __init__.py
│   ├── inference_engine.py      # Smart dual inference system
│   └── model_utils.py          # CNN model utilities
├── notebooks/                   # Jupyter notebooks
│   ├── baseline_cnn.ipynb      # CNN training notebook
│   └── model_integration.ipynb  # Integration testing
├── models/                      # Model files
│   ├── unified_model.h5        # Trained CNN model
│   └── screenrecognition-web350k-vins.torchscript
├── metadata/                    # Model metadata
│   └── screenrecognition/
│       └── class_dict.json
├── examples/                    # Sample images
│   └── sample_images/
│       ├── sketch-example.jpg
│       └── ui-screenshot.jpg
├── utils/                       # Utilities
│   └── data_downloader.py
└── requirements.txt            # Project dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ProPain03/SketchReal.git
   cd SketchReal
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models** (if not included)
   - Place `unified_model.h5` in `models/` directory
   - Place `screenrecognition-web350k-vins.torchscript` in `models/` directory

### Basic Usage

#### 1. Using the Inference Engine

```python
from config.config import Config
from src.inference_engine import SmartDualInference

# Initialize system
config = Config()
smart_system = SmartDualInference(
    unified_model_path=config.UNIFIED_MODEL_PATH,
    screenrec_model_path=config.SCREENREC_MODEL_PATH,
    config=config
)

# Run inference
results = smart_system.inference_pipeline(
    image_path="path/to/your/image.jpg",
    input_type=None,  # Auto-detect
    sketch_conf=0.4,
    screenrec_conf=0.5,
    enable_fragment_filtering=True
)

# Access results
detections = results['fused_results']
for detection in detections:
    bbox = detection['bbox']
    class_name = detection['class_name']
    confidence = detection['confidence']
    print(f"{class_name}: {confidence:.2f} at {bbox}")
```

#### 2. Using the Web Interface

```bash
cd app
streamlit run app.py
```

## 🧠 Core Features

### Smart Input Detection
- Automatically distinguishes between hand-drawn sketches and UI screenshots
- Adapts processing pipeline based on input type

### Dual Model Architecture
- **Sketch Mode**: Prioritizes unified CNN for hand-drawn elements
- **UI Mode**: Prioritizes ScreenRecognition for real UI components

### Advanced Filtering
- **Fragment Filtering**: Removes character-level noise
- **Duplicate Suppression**: Eliminates redundant detections
- **Text Merging**: Combines fragmented text elements
- **Confidence Boosting**: Enhances aligned detections

### Fusion Strategies
- **QuickDraw Priority**: For sketch inputs
- **ScreenRec Priority**: For UI screenshots (strict mode)
- **ScreenRec Only**: Pure UI detection mode

## 📊 Supported UI Classes

The system can detect 12 different UI element types:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Other | Miscellaneous elements |
| 1 | Background Image | Background images/wallpapers |
| 2 | Checked View | Checkboxes, radio buttons |
| 3 | Icon | Application icons, symbols |
| 4 | Input Field | Text input boxes |
| 5 | Image | Pictures, photos |
| 6 | Text | Text labels, content |
| 7 | Text Button | Clickable text buttons |
| 8 | Page Indicator | Navigation dots, pagination |
| 9 | Pop-Up Window | Modal dialogs, alerts |
| 10 | Sliding Menu | Drawer menus, sidebars |
| 11 | Switch | Toggle switches |

## 🔧 Configuration

Modify `config/config.py` to customize:

```python
class Config:
    # Model paths
    UNIFIED_MODEL_PATH = "models/unified_model.h5"
    SCREENREC_MODEL_PATH = "models/screenrecognition-web350k-vins.torchscript"
    
    # Detection parameters
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    BATCH_SIZE = 32
    
    # Class mappings
    UI_LABELS = {
        0: 'Other', 1: 'Background Image', 2: 'Checked View',
        # ... etc
    }
```

## 📈 Performance Optimization

### Inference Parameters

- `sketch_conf`: Confidence threshold for sketch detection (0.0-1.0)
- `screenrec_conf`: Confidence threshold for UI detection (0.0-1.0)
- `iou_threshold`: IoU threshold for fusion (0.0-1.0)
- `strict_ui_mode`: Enable strict filtering for UI images
- `enable_fragment_filtering`: Enable advanced fragment filtering

### Speed vs Accuracy Trade-offs

```python
# Fast inference (lower accuracy)
results = smart_system.inference_pipeline(
    image_path="image.jpg",
    sketch_conf=0.6,
    screenrec_conf=0.7,
    enable_fragment_filtering=False
)

# High accuracy (slower)
results = smart_system.inference_pipeline(
    image_path="image.jpg",
    sketch_conf=0.3,
    screenrec_conf=0.4,
    enable_fragment_filtering=True,
    strict_ui_mode=True
)
```

## 🧪 Testing and Development

### Running Tests

```bash
# Test individual components
python -c "
from src.inference_engine import SmartDualInference
from config.config import Config
config = Config()
system = SmartDualInference('', '', config)
print('✅ Components working')
"
```

### Development Notebooks

1. **baseline_cnn.ipynb**: Train and evaluate the unified CNN model
2. **model_integration.ipynb**: Test the complete integration system

### Adding Sample Images

Place test images in `examples/sample_images/`:
- UI screenshots: `.jpg`, `.png` files of mobile/web interfaces
- Sketches: Hand-drawn UI mockups

### Performance Issues

- **Memory**: Reduce batch size in config
- **Speed**: Enable quantization, reduce image resolution
- **Accuracy**: Increase confidence thresholds, enable strict mode

## 📋 Dependencies

### Core Dependencies
- `tensorflow>=2.8.0`: Deep learning framework
- `torch>=1.10.0`: PyTorch for ScreenRecognition
- `torchvision>=0.11.0`: Computer vision utilities
- `opencv-python>=4.5.0`: Image processing
- `numpy>=1.21.0`: Numerical computing
- `pillow>=8.3.0`: Image handling

### Web Interface
- `streamlit>=1.15.0`: Web framework
- `matplotlib>=3.5.0`: Visualization
- `plotly>=5.0.0`: Interactive plots

### Development
- `jupyter>=1.0.0`: Notebook environment
- `scikit-learn>=1.0.0`: Machine learning utilities