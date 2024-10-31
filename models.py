from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
def load_model_and_processor_apple_model():
    """
    Loads and returns the MobileViTV2 model and feature extractor https://huggingface.co/apple/mobilevitv2-1.0-imagenet1k-256
    
    Returns:
        tuple: (feature_extractor, model) pair for image classification
    """
    feature_extractor = MobileViTImageProcessor.from_pretrained("shehan97/mobilevitv2-1.0-imagenet1k-256")
    model = MobileViTV2ForImageClassification.from_pretrained("shehan97/mobilevitv2-1.0-imagenet1k-256")
    return feature_extractor, model

def predict_image_class_apple_model(path_to_image, feature_extractor, model):
    """
    Predicts the ImageNet class for a given PIL Image
    
    Args:
        image (PIL.Image): Input image to classify
        feature_extractor: MobileViT feature extractor
        model: MobileViTV2 classification model
        
    Returns:
        str: Predicted class label
    """
    image = Image.open(path_to_image).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    
    return model.config.id2label[predicted_class_idx]