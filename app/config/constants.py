# Image Processing Constants
TARGET_IMAGE_SIZE = (512, 512)
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']

# Model Constants
MODEL_REGISTRY = {
    'mobilevit_v2': 'MobileViTClassifier',
    'openai': 'OpenAIImageClassifier'
}

# OpenAI Constants
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# API Response Messages
ERROR_MESSAGES = {
    'NO_IMAGE': 'No image file provided',
    'NO_FILE_SELECTED': 'No selected file',
    'NO_HTML_CONTENT': 'No HTML content provided',
    'INVALID_MODEL': lambda available: f"Model not found. Available models: {available}",
    'ENV_ERROR': 'OPENAI_API_KEY is not set or empty in the environment variables'
}

# Default Processing Options
DEFAULT_OUTPUT_TYPE = 'detailed'
DEFAULT_DB_LIMIT = 250