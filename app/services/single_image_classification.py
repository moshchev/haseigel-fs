from PIL import Image
from ..core.image_models import MobileViTClassifier, OpenAIImageClassifier

MODEL_REGISTRY = {
    'mobilevit_v2': MobileViTClassifier(),
    # 'openai': OpenAIImageClassifier()
}

def classify_image(image_file, model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Load and preprocess the image
    image = Image.open(image_file)
    
    # Get the appropriate model
    model = MODEL_REGISTRY[model_name]
    
    # Perform classification
    result = model.predict(image)
    
    return result