import os

class Config:
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Models and metadata directories
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    METADATA_DIR = os.path.join(PROJECT_ROOT, 'metadata')

    # Metadata paths
    CLASS_DICT_PATH = os.path.join(METADATA_DIR, "screenrecognition", "class_dict.json")
    CLASS_MAP_PATH = os.path.join(METADATA_DIR, "screenrecognition", "class_map_vins_manual.json")

    # Model paths
    UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, "unified_model.h5")
    SCREENREC_MODEL_PATH = os.path.join(MODEL_DIR, "screenrecognition-web350k-vins.torchscript")
    
    # Data parameters
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    NUM_CLASSES = 12  # Updated for new UI labels
    
    # Training parameters
    EPOCHS = 15  # Increased for better convergence
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1

    # QuickDraw categories
    QUICKDRAW_CATEGORIES = [
        "star", "envelope", "house", "gear", "arrow", "plus", "triangle", "circle", "square", "heart",
        "check", "x", "mountain", "tree", "sun", "moon", "cloud", "rectangle", "line",
        "dot", "frame", "light switch", "power outlet", "scribble", "zigzag", "abstract"
    ]

    # Mapping from QuickDraw categories to UI labels
    qd_to_ui_label = {
        "star": "Icon",
        "envelope": "Icon",
        "house": "Icon",
        "gear": "Icon",
        "arrow": "Icon",
        "plus": "Icon",
        "triangle": "Icon",
        "circle": "Icon",
        "square": "Icon",
        "heart": "Icon",
        "check": "Checked View",
        "x": "Checked View",
        "mountain": "Background Image",
        "tree": "Background Image",
        "sun": "Background Image",
        "moon": "Background Image",
        "cloud": "Other",
        "rectangle": "Input Field",
        "line": "Input Field",
        "dot": "page_indicator",
        "frame": "Pop-Up Window",
        "light Switch": "Switch",
        "power outlet": "Switch",
        "scribble": "Other",
        "zigzag": "Other",
        "abstract": "Other"
    }

    # Final UI Labels for classification
    UI_LABELS = {
        0: 'Other',              
        1: 'Background Image',   
        2: 'Checked View',      
        3: 'Icon',             
        4: 'Input Field',      
        5: 'Image',            
        6: 'Text',             
        7: 'Text Button',       
        8: 'Page Indicator',   
        9: 'Pop-Up Window',     
        10: 'Sliding Menu',     
        11: 'Switch'          
    }

    # Reverse mapping for easy lookup
    LABEL_TO_ID = {v: k for k, v in UI_LABELS.items()}

    # Synthetic data configuration
    SAMPLES_PER_CLASS = 2500  # Equal sampling for class balance
    SYNTHETIC_SAMPLES_PER_LABEL = 300  # Additional synthetic samples per UI label