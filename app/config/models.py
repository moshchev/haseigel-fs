from app.core.image_models import MobileViTClassifier, MoondreamProcessor, AsyncVisionLanguageModelClassifier

# this dict is used for a lazy load for a single image classification endpoint

MODEL_CLASSES = {
    'mobilevit_v2': MobileViTClassifier,
    'moondream': MoondreamProcessor,
    'vllm': AsyncVisionLanguageModelClassifier,
}