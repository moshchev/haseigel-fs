from pydantic import Field, create_model, BaseModel
from typing import Optional

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