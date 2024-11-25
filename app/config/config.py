import os
from pathlib import Path
from . import DEFAULT_MODEL_NAME, MODEL_REGISTRY

# Get base directory of project
BASE_DIR = Path(__file__).resolve().parent.parent

# Image directories
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'images')
TEMP_IMAGE_DIR = os.path.join(IMAGE_DIR, 'temp')

# Ensure directories exist
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)