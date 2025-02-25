"""
Configuration settings for the SignSpeak application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Model Settings
SIGN_RECOGNITION_MODEL_PATH = os.getenv('SIGN_RECOGNITION_MODEL_PATH', 'src/models/sign_recognition_model.pth')
USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'

# Webcam Settings
WEBCAM_INDEX = int(os.getenv('WEBCAM_INDEX', '0'))

# Application Settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# UI Settings
STREAMLIT_SERVER_PORT = int(os.getenv('STREAMLIT_SERVER_PORT', '8501'))
FLASK_SERVER_PORT = int(os.getenv('FLASK_SERVER_PORT', '5000'))

# MediaPipe Settings
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0=Lite, 1=Full, 2=Heavy
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Translation Settings
MAX_TRANSLATION_TOKENS = 50
TRANSLATION_TEMPERATURE = 0.7

# Dataset Paths
TRAINING_DATA_PATH = os.path.join('src', 'data', 'training')
VALIDATION_DATA_PATH = os.path.join('src', 'data', 'validation')
TEST_DATA_PATH = os.path.join('src', 'data', 'test')

# Create necessary directories if they don't exist
for path in [TRAINING_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH]:
    os.makedirs(path, exist_ok=True) 