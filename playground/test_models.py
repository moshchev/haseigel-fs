import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import asyncio

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.image_models import MoondreamProcessor

image_path = 'data/images/temp/Wintergrillen 992x661.jpg.webp'
img = Image.open(image_path)

categories = ["grill", "axe", "hammer"]
cats = []

moondream = MoondreamProcessor()
results = asyncio.run(moondream.process_single_image(img, categories))

print(results)