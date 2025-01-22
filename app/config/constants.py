# Image Processing Constants
TARGET_IMAGE_SIZE = (512, 512)
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']

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

# there is some stuff in the code that is hardcoded, you can add it here (as inspiration)