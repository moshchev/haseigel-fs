from typing import List, Dict, Any
import asyncio
from app.utils.data_tool import get_html_data_as_json, create_db_engine
from dotenv import load_dotenv
from app.core.image_models import MobileViTClassifier, AsyncVisionLanguageModelClassifier
from app.config import TEMP_IMAGE_DIR
from app.loaders import ImageLoader
from app.services.extract_images import download_images, collect_image_data


class ModelLoader:
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


async def process_images_local(categories:list[str]=None, batch_size: int = 8):
    """
    Main processing function that sets up and runs the producer-consumer pattern.
    """
    # Initialize image loader
    image_loader = ImageLoader(TEMP_IMAGE_DIR, target_size=(512, 512))
    
    # Initialize queue
    queue = asyncio.Queue()
    
    # Initialize processor
    processor = ModelLoader(model_type="local")
    # Create tasks
    producer_task = asyncio.create_task(producer_local(image_loader, queue, batch_size))
    consumer_task = asyncio.create_task(consumer_local(queue, processor.model))
    
    # Wait for both tasks to complete
    await producer_task
    results = await consumer_task
    
    return results


def get_data_list(categories: List[str]):
    assert load_dotenv()
    engine = create_db_engine()
    html_data_list = get_html_data_as_json(engine)
    
    return html_data_list, categories


async def process_images_hosted(data_list: List[Dict[str, Any]], categories: List[str]):
    
    # Collect and download images
    image_data = collect_image_data(data_list['data'])
    downloaded_images = download_images(image_data, TEMP_IMAGE_DIR)
    
    # Process images with specified model
    model = ModelLoader(model_type="hosted")
    image_paths = [img["local_path"] for img in downloaded_images]
    results = await model.model.predict_batch(image_paths, categories=categories)

    return results

if __name__ == "__main__":
    asyncio.run(process_images_hosted())

