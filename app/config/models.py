from app.core.image_models import MobileViTClassifier, MoondreamProcessor, AsyncVisionLanguageModelClassifier

MODEL_CLASSES = {
    'mobilevit_v2': MobileViTClassifier,
    'moondream': MoondreamProcessor,
    'vllm': AsyncVisionLanguageModelClassifier,
}