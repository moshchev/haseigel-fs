from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os
from ..utils import prepare_image, create_dynamic_schema, ImagePrompt

class MobileViTClassifier:
    def __init__(self):
        self.feature_extractor, self.model = self._load_model_and_processor()
    
    def _load_model_and_processor(self):
        """
        Loads and returns the MobileViTV2 model and feature extractor
        """
        feature_extractor = MobileViTImageProcessor.from_pretrained("shehan97/mobilevitv2-1.0-imagenet1k-256")
        model = MobileViTV2ForImageClassification.from_pretrained("shehan97/mobilevitv2-1.0-imagenet1k-256")
        return feature_extractor, model
    
    def predict(self, image):
        """
        Predicts the ImageNet class for a given PIL Image
        
        Args:
            image (PIL.Image): Input image to classify
            
        Returns:
            dict: Prediction results including class label
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
            
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_class_idx]
        
        return {
            "prediction": predicted_class,
            "model": "mobilevit_v2"
        }
    
class OpenAIImageClassifier():
    def __init__(self, model_name:str="gpt-4o-mini"):
        load_dotenv()
        self._validate_environment()
        self.client = OpenAI()
        self.model_name = model_name

    def _validate_environment(self) -> None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:  # This checks for None or empty string
            raise EnvironmentError("OPENAI_API_KEY is not set or empty in the environment variables")

    def _prepare_message(self, image_path:str, prompt:str) -> list[dict]:
        base64_image = prepare_image(image_path)
        message = [
            {"role": "user", 
            "content": [
                {"type": "text", "text": prompt}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
            }
        ]
        return message

    def predict(self, image_path:str, categories:list[str]) -> dict:
        prompt = ImagePrompt.DEFAULT_PROMPT
        schema = create_dynamic_schema(categories)
        message = self._prepare_message(image_path, prompt)
        # beta.chat.completions.parse
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            response_format=schema
        )
        try:
            response_data = response.model_dump()
            schema.parse_obj(response_data)  # Validate the response against the schema
        except Exception as e:
            raise ValueError(f"Response validation failed: {e}")
        
        return response_data