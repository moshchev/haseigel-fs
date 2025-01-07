from collections import defaultdict
from typing import List, Dict, Any, Tuple
import asyncio
from PIL import Image
from app.services.extract_images import extract_img_attributes, download_images_with_local_path
from app.utils.data_tool import get_html_data_as_json, create_db_engine
from dotenv import load_dotenv
from app.core.image_models import MobileViTClassifier, AsyncVisionLanguageModelClassifier
from urllib.parse import urlparse
from app.config import TEMP_IMAGE_DIR
from tests.moondream_transformers import ImageLoader
import os

class ImageProcessor:
    def __init__(self, model_type: str = "local"):
        """
        Initialize the image processor with specified model type.
        
        Args:
            model_type: "local" or "hosted" 
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        if self.model_type == "local":
            return MobileViTClassifier()
        elif self.model_type == "hosted":
            return AsyncVisionLanguageModelClassifier()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")


async def producer_local(image_loader: ImageLoader, queue: asyncio.Queue, batch_size: int = 8):
    """
    Produces batches of images from the ImageLoader.
    """
    image_loader = ImageLoader(folder_path=TEMP_IMAGE_DIR, target_size=(512, 512), max_workers=8)
    for batch in image_loader.batch_images(batch_size):
        await queue.put(batch)
    # Signal completion
    await queue.put(None)


async def consumer_local(queue: asyncio.Queue, model):
    """
    Consumes image batches and processes them with local model.
    """
    results = []
    while True:
        batch = await queue.get()
        if batch is None:
            break
            
        filenames, images = batch
        try:
            # Process batch with local model
            for filename, img in zip(filenames, images):
                prediction = model.predict(img)['prediction']
                results.append({
                    "image_path": filename,
                    "predicted_class": prediction
                })
        except Exception as e:
            print(f"Error processing batch: {e}")
        finally:
            queue.task_done()
    return results

async def consumer_vllm(queue: asyncio.Queue, model: AsyncVisionLanguageModelClassifier, categories: List[str] = None):
    """
    Consumes batches of image paths and processes them with VLLM model.
    """
    results = []
    while True:
        batch = await queue.get()
        if batch is None:
            break
            
        try:
            # Process entire batch at once
            predictions = await model.predict_batch(batch, categories)
            
            # Store results
            for path, prediction in zip(batch, predictions):
                results.append({
                    "image_path": path,
                    "predicted_class": prediction
                })
        except Exception as e:
            print(f"Error processing batch: {e}")
        finally:
            queue.task_done()
    return results


async def process_images(categories:list[str]=None, model_type: str = "hosted", batch_size: int = 8):
    """
    Main processing function that sets up and runs the producer-consumer pattern.
    """
    # Initialize image loader
    image_loader = ImageLoader(TEMP_IMAGE_DIR, target_size=(512, 512))
    
    # Initialize queue
    queue = asyncio.Queue()
    
    # Initialize processor
    processor = ImageProcessor(model_type)
    
    # Create tasks
    
    if model_type == "local":
        producer_task = asyncio.create_task(producer_local(image_loader, queue, batch_size))
        consumer_task = asyncio.create_task(consumer_local(queue, processor.model))
    elif model_type == "hosted":
        producer_task = asyncio.create_task(producer_vllm(image_loader, queue, batch_size))
        consumer_task = asyncio.create_task(consumer_vllm(queue, processor.model, categories))
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Wait for both tasks to complete
    await producer_task
    results = await consumer_task
    
    return results

def collect_image_data(html_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collects image data from multiple HTML and organizes it by domain ID.
    Filters images to include only .jpeg, .jpg, .png extensions and excludes 'logo' in filename.
    
    Args:
        html_data_list: List of dictionaries containing:
            - domain_start_id: ID of the domain
            - base_url: Base URL for the domain
            - response_text: HTML content
        
    Returns:
        List of dictionaries with structure: 
        [{'domain_id': domain_id, 'images': ['http://...', 'http://...']}]
    """
    domain_images = {}
    valid_extensions = ('.jpeg', '.jpg', '.png')
    
    for data in html_data_list:
        domain_id = data['domain_start_id']
        base_url = data['base_url'][0]
        html = data['response_text'][0]
        
        # Get image data for this HTML
        img_data = extract_img_attributes(html, base_url)
        
        # Extract URLs and filter by extension and exclude logos
        image_urls = [
            img.get('src') for img in img_data 
            if img.get('src') and (
                any(img.get('src').lower().endswith(ext) for ext in valid_extensions) and
                'logo' not in img.get('src').lower()
            )
        ]
        
        # Add to domain_images dict
        if domain_id in domain_images:
            domain_images[domain_id].extend(image_urls)
            # Remove duplicates while preserving order
            domain_images[domain_id] = list(dict.fromkeys(domain_images[domain_id]))
        else:
            domain_images[domain_id] = image_urls
    
    # Convert to final format
    result = [
        {'domain_id': domain_id, 'images': images}
        for domain_id, images in domain_images.items()
    ]
    
    return result

def download_images(image_data: List[Dict[str, List[str]]], temp_dir: str) -> List[Dict[str, Any]]:
    """
    Downloads all images from the collected image data.
    
    Args:
        image_data: List of dictionaries with structure {'domain_id': id, 'images': [urls]}
        temp_dir: Directory to store downloaded images
        
    Returns:
        List of dictionaries containing downloaded image information
    """
    downloaded_images = []
    
    for domain_data in image_data:
        domain_id = domain_data['domain_id']
        image_urls = domain_data['images']
        
        # Create list of dicts with URLs and domain_id
        images_to_download = [
            {'src': url, 'domain_id': domain_id}
            for url in image_urls
        ]
        
        # Download images with domain-specific names
        download_images_with_local_path(images_to_download, temp_dir)
        
        # Filter out failed downloads
        downloaded_images.extend([img for img in images_to_download if img.get("local_path")])
    
    return downloaded_images

# def organize_predictions_by_domain(predictions: List[Dict[str, Any]], downloaded_images: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Organizes predictions by domain ID using the downloaded images data.
    
#     Args:
#         predictions: List of dictionaries with structure {"file_path": str, "prediction": dict}
#         downloaded_images: List of dictionaries from download_images function containing domain_id and local_path
        
#     Returns:
#         Dictionary mapping domain IDs to lists of predictions:
#         {
#             "domain_id": [
#                 {"file_path": str, "prediction": dict},
#                 ...
#             ]
#         }
#     """
#     # Create a mapping of local_path to domain_id
#     path_to_domain = {img["local_path"]: img["domain_id"] for img in downloaded_images}
    
#     # Initialize results dictionary
#     domain_predictions = defaultdict(list)
    
#     # Organize predictions by domain
#     for pred in predictions:
#         file_path = pred["file_path"]
#         if file_path in path_to_domain:
#             domain_id = path_to_domain[file_path]
#             domain_predictions[domain_id].append(pred)
    
#     return dict(domain_predictions)

async def main():
    assert load_dotenv()
    engine = create_db_engine()
    html_data_list = get_html_data_as_json(engine)
    
    # Collect and download images
    image_data = collect_image_data(html_data_list['data'])
    downloaded_images = download_images(image_data, TEMP_IMAGE_DIR)
    
    # Process images with specified model
    processor = ImageProcessor(model_type="hosted")
    image_paths = [img["local_path"] for img in downloaded_images]
    results = await processor.model.predict_batch(image_paths, categories=["hammer", "nail", "tape"])
    print(results)
    return results

if __name__ == "__main__":
    asyncio.run(main())
    # assert load_dotenv()
    # engine = create_db_engine()
    # html_data_list = get_html_data_as_json(engine)
    
    # # Collect and download images
    # image_data = collect_image_data(html_data_list['data'])
    # downloaded_images = download_images(image_data, TEMP_IMAGE_DIR)
    # print(downloaded_images)
    # loader =ImageLoader(TEMP_IMAGE_DIR, target_size=(512, 512), max_workers=8)
    # loader.load_images()
    # for batch in loader.batch_images(batch_size=8):
    #     print(batch)
