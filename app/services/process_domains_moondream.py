import asyncio

from app.services.extract_images import collect_image_data, download_images
from app.core.image_models import MoondreamProcessor
from app.loaders import ImageLoader

from app.config import TEMP_IMAGE_DIR

import time
# Producer: Loads image batches and sends them to the queue
async def producer(image_loader, batch_size, queue):
    for batch in image_loader.batch_images(batch_size):
        await queue.put(batch)
    # Signal that we're done
    await queue.put(None)

# Consumer: Pulls batches from the queue and runs model inference
async def consumer(queue, moondream_processor, categories):
    while True:
        batch = await queue.get()
        if batch is None:
            # No more data, exit
            break
        # Run the async method directly (no need for run_in_executor)
        results = await moondream_processor.process_batch(batch, categories)

        # Do whatever you want with results
        for filename, answers in results.items():
            print(f"Results for {filename}: {answers}")

# The main entry point tying it all together
async def process_domains_moondream(image_loader, moondream_processor, categories, batch_size=2):
    # Create an asyncio queue
    q = asyncio.Queue()

    # Create the producer and consumer tasks
    prod_task = asyncio.create_task(producer(image_loader, batch_size, q))
    cons_task = asyncio.create_task(consumer(q, moondream_processor, categories))

    # Wait until both are done
    await asyncio.gather(prod_task, cons_task)


def process_domains_moondream_service(data, categories):
    """
    data: List[Dict[str, Any]]
    categories: List[str]
    """
    # Collect and download images
    image_data = collect_image_data(data)
    download_images(image_data, TEMP_IMAGE_DIR)
    
    # Initialize image loader and processor
    image_loader = ImageLoader(folder_path=TEMP_IMAGE_DIR, target_size=(512, 512), max_workers=8)
    moondream = MoondreamProcessor()

    # Launch the async pipeline
    results = asyncio.run(process_domains_moondream(image_loader, moondream, categories, batch_size=2))

    return results
