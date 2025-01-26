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
    # Initialize statistics
    stats = {
        'total_images': 0,
        'categories': {category: 0 for category in categories},
        'per_route': {}
    }
    
    while True:
        batch = await queue.get()
        if batch is None:
            # No more data, exit
            break
            
        # Run the async method directly (no need for run_in_executor)
        results = await moondream_processor.process_batch(batch, categories)

        # Update statistics
        for filename, answers in results.items():
            stats['total_images'] += 1
            
            # Extract route information from filename if available
            route = filename.split('_')[0]  # Assuming route is first part of filename
            if route not in stats['per_route']:
                stats['per_route'][route] = {
                    'count': 0,
                    'categories': {category: 0 for category in categories}
                }
            stats['per_route'][route]['count'] += 1
            
            # Update category counts
            for category, answer in answers.items():
                if answer:  # If answer is True/positive
                    stats['categories'][category] += 1
                    stats['per_route'][route]['categories'][category] += 1
                    
    # Return final statistics
    return stats

# The main entry point tying it all together
async def process_domains_moondream(image_loader, moondream_processor, categories, batch_size=2):
    # Create an asyncio queue
    q = asyncio.Queue()

    # Create the producer and consumer tasks
    prod_task = asyncio.create_task(producer(image_loader, batch_size, q))
    cons_task = asyncio.create_task(consumer(q, moondream_processor, categories))

    # Wait until both are done and get final statistics
    await prod_task
    stats = await cons_task
    
    # Enhanced statistics output
    print("\n=== Processing Summary ===")
    print(f"\nTotal images processed: {stats['total_images']}")
    
    # Calculate percentages
    total_images = stats['total_images']
    if total_images > 0:
        print("\nOverall Classification Rates:")
        for category, count in stats['categories'].items():
            percentage = (count / total_images) * 100
            print(f"- {category.title()}: {count} ({percentage:.1f}%)")
        
        # Find most and least common categories
        sorted_categories = sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_categories[0]
        least_common = sorted_categories[-1]
        
        print(f"\nMost common category: {most_common[0].title()} ({most_common[1]} occurrences)")
        print(f"Least common category: {least_common[0].title()} ({least_common[1]} occurrences)")
    
    print("\n=== Route Breakdown ===")
    for route, data in stats['per_route'].items():
        print(f"\nRoute {route}:")
        print(f"- Total images: {data['count']}")
        print("- Classification Rates:")
        for category, count in data['categories'].items():
            percentage = (count / data['count']) * 100 if data['count'] > 0 else 0
            print(f"  {category.title()}: {count} ({percentage:.1f}%)")
    
    return stats


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
