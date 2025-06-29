import os
def __init__(self, config_path=None):
    if config_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        unified_model_path = os.path.join(project_root, "models", "unified_model.h5")
        screenrec_model_path = os.path.join(project_root, "models", "screenrecognition-web350k-vins.torchscript")
        metadata_path = os.path.join(project_root, "metadata", "screenrecognition")