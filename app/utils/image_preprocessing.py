from PIL import Image
import base64
import logging

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocesses an image by resizing and converting it to grayscale.
    """
    try:
        img = Image.open(image_path)
        img = img.resize(target_size, Image.LANCZOS).convert("L")
        return img
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None


def encode_image_to_base64(image_path):
    """
    Encodes an image to a base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_path} to base64: {e}")
        return None