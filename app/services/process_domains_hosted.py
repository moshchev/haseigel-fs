from app.loaders import ModelLoader
from app.utils import collect_image_data, download_images
from app.config import TEMP_IMAGE_DIR
from typing import List, Dict, Any

async def process_images_hosted(data_list: List[Dict[str, Any]], categories: List[str]):
    
    # Collect and download images
    image_data = collect_image_data(data_list['data'])
    downloaded_images = download_images(image_data, TEMP_IMAGE_DIR)
    
    # Process images with specified model
    model = ModelLoader(model_type="hosted")
    image_paths = [img["local_path"] for img in downloaded_images]
    results = await model.model.predict_batch(image_paths, categories=categories)

    return results