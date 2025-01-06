from app.core.image_models import MobileViTClassifier, OpenAIImageClassifier, MoondreamProcessor

MODEL_CLASSES = {
    'mobilevit_v2': MobileViTClassifier,
    'openai': OpenAIImageClassifier,
    'moondream': MoondreamProcessor,
}