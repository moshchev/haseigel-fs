# from dotenv import load_dotenv
import os
import sys
import moondream as md
from PIL import Image

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load_dotenv()

# Initialize with API key
model = md.vl(api_key=os.environ['MOONDREAM'])

# Load an image
img = '/Users/alexander/Desktop/projects/haseigel-fs/data/images/temp/Wintergrillen 992x661.jpg.webp'
image = Image.open(img)
encoded_image = model.encode_image(image)  # Encode image (recommended for multiple operations)

# Detect objects
detect_result = model.detect(image, 'grill')  # change 'subject' to what you want to detect
print("Detected objects:", detect_result["objects"])