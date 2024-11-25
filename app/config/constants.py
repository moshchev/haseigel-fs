from pydantic import Field, create_model, BaseModel
from typing import Optional


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

# Image Prompts
class ImagePrompts:
    DEFAULT_PROMPT = """
    You are an advanced AI system specializing in image recognition. 
    Your task is to analyze the provided image and determine if it contains any of the categories provided in the schema. 
    **Instructions:**
    - For each category, respond with `true` if the object is clearly visible, and `false` otherwise.
    - If none of the categories match the content of the image, assign your own single-word category to describe the image.
    """


    NO_CATEGORIES_PROMPT = """
    You are an advanced AI system specializing in image recognition.
    Your task is to analyze the provided image and determine the most appropriate category for the image.
    **Instructions:**
    - Analyze the image and provide a single category that best describes the content of the image.
    """


class NoCategoriesSchema(BaseModel):
    prediction: str


def create_dynamic_schema(categories: list[str]):
    """
    Dynamically create a Pydantic model schema based on the user-provided categories.
    """
    fields = {
        category: (bool, Field(..., description=f"Whether the image contains {category}"))
        for category in categories
    }
    fields['custom_category'] = (Optional[str], str)
    return create_model("DynamicImageSchema", **fields)

