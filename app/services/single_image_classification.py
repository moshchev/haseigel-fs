from app.config.models import MODEL_CLASSES

MODEL_REGISTRY = {model_name: None for model_name in MODEL_CLASSES}

def get_model(model_name: str):
    """Lazy initialization of models"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    if MODEL_REGISTRY[model_name] is None:
        MODEL_REGISTRY[model_name] = MODEL_CLASSES[model_name]()
    
    return MODEL_REGISTRY[model_name]


def classify_image(image_file, model_name: str):
    # Get the appropriate model
    model = get_model(model_name)
    
    # Perform classification
    result = model.predict(image_file)
    
    return result