from moondream_transformers import ImageLoader, MoondreamProcessor
import asyncio
import time

# Producer: Loads image batches and sends them to the queue
async def producer(image_loader, batch_size, queue):
    for batch in image_loader.batch_images(batch_size):
        await queue.put(batch)
    # Signal that we're done
    await queue.put(None)

# Consumer: Pulls batches from the queue and runs model inference
async def consumer(queue, moondream_processor, queries):
    while True:
        batch = await queue.get()
        if batch is None:
            # No more data, exit
            break
        # Run the async method directly (no need for run_in_executor)
        results = await moondream_processor.process_batch(batch, queries)

        # Do whatever you want with results
        for filename, answers in results.items():
            print(f"Results for {filename}: {answers}")

# The main entry point tying it all together
async def main(image_loader, moondream_processor, queries, batch_size=2):
    # Create an asyncio queue
    q = asyncio.Queue()

    # Create the producer and consumer tasks
    prod_task = asyncio.create_task(producer(image_loader, batch_size, q))
    cons_task = asyncio.create_task(consumer(q, moondream_processor, queries))

    # Wait until both are done
    await asyncio.gather(prod_task, cons_task)

if __name__ == "__main__":
    start_time = time.time()
    # Your existing setup
    image_loader = ImageLoader(folder_path="data/images/test_set", target_size=(512, 512), max_workers=8)
    moondream = MoondreamProcessor()
    queries = [
        "is there a grill in this image? answer yes or no",
        "is there an axe in this image? answer yes or no",
        "is there a chair in this image? answer yes or no"
    ]

    # Launch the async pipeline
    asyncio.run(main(image_loader, moondream, queries, batch_size=2))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds") 

