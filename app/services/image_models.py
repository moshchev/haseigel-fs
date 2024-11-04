from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image

class MobileViTClassifier:
    def __init__(self):
        self.feature_extractor, self.model = self.load_model_and_processor()
    
    def load_model_and_processor(self):
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