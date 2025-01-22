from PIL import Image
import asyncio

import os
import sys
# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.core.image_models import VisionLanguageModelClassifier, MoondreamProcessor

image_path = 'data/images/temp/Wintergrillen 992x661.jpg.webp'
categories = ["grill", "axe", "hammer"]


def test_vision_language_model_classifier():
    model = VisionLanguageModelClassifier()
    prediction = model.predict(image_path, categories)
    print(prediction)

def test_moondream_classifier():
    model = MoondreamProcessor()
    prediction = asyncio.run(model.process_single_image(Image.open(image_path), categories))
    print(prediction)


if __name__ == "__main__":
    test_moondream_classifier()