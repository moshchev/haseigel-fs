import os
import sys
# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.core.image_models import OpenAIImageClassifier
from app.config import DEFAULT_MODEL_NAME, ERROR_MESSAGES

image_path = 'data/images/temp/0_1156EF.jpeg-200x200.jpg'

model = OpenAIImageClassifier()
# categories = ["cat", "dog", "bird"]
prediction = model.predict(image_path)
print(prediction)