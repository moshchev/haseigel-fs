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


    @classmethod
    def get_categorized_prompt(cls, categories: list[str]) -> str:
        categories_str = ", ".join(categories)
        return f"""
        You are an advanced AI system specializing in image recognition. Your task is to analyze the provided image and determine if it contains any of the following categories: {categories_str}.
        
        **Instructions:**
        - For each category, respond with `true` you can spot the object in the image, and `false` otherwise.
        - Make sure that you check the presence of the object in the image. Do it for each category, and think step by step. -> I should look carefully at the image and see if the object is present. Do this for each category.
        - If none of the categories match the content of the image, assign your own single-word category to describe the image.
        - Return your analysis in the following JSON format:
        ```json
        {{
            "categories": {{
                "category1": true/false,
                "category2": true/false,
                ...
            }},
            "custom_category": "your-category-here" (if applicable)
        }}
        ```
        - Do not include any explanations or text outside of the JSON object.
        - Ensure the JSON is properly formatted with no syntax errors.
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