# from transformers import AutoModelForCausalLM, AutoTokenizer
# from PIL import Image
# import time
# import torch

# def moondream_transformers():
#     start_time = time.time()

#     model_id = "vikhyatk/moondream2"
#     revision = "2024-08-26"
#     # Add device detection
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, trust_remote_code=True, revision=revision
#     ).to(device)  # Move model to GPU
#     tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

#     model_load_end = time.time()

#     print(f"Using device: {device}")  # Print which device is being used
#     print(f"Model loaded in {model_load_end - start_time:.2f} seconds")

#     # Image processing time
#     image_process_start = time.time()

#     image = Image.open('data/images/temp/Wintergrillen 992x661.jpg.webp')
#     enc_image = model.encode_image(image)
#     image_process_end = time.time()
#     print(f"Image processed in {image_process_end - image_process_start:.2f} seconds")

#     query_start = time.time()
#     answer = model.answer_question(enc_image, "is there a grill in this image? answer yes or no", tokenizer)
#     query_end = time.time()
#     print(f"Query 1 answered in {query_end - query_start:.2f} seconds")

#     query_start = time.time()
#     answer = model.answer_question(enc_image, "is there a axe in this image? answer yes or no", tokenizer)
#     query_end = time.time()
#     print(f"Query 2 answered in {query_end - query_start:.2f} seconds")

#     query_start = time.time()
#     answer = model.answer_question(enc_image, "is there a chair in this image? answer yes or no", tokenizer)
#     query_end = time.time()
#     print(f"Query 3 answered in {query_end - query_start:.2f} seconds")
#     # print(answer)


#     end_time = time.time()
#     print(f"Total time: {end_time - start_time:.2f} seconds")

import time
import torch
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time


class MoondreamProcessor:
    def __init__(self, model_id="vikhyatk/moondream2", revision="2024-08-26"):
        """Initialize the model once and load it into memory."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer once
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    async def encode_image_async(self, image):
        """Encodes a single image asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.model.encode_image, image)

    async def encode_images(self, batch):
        """Encodes a batch of images asynchronously."""
        encode_start = time.time()
        
        # Unpack the batch tuple correctly
        filenames, images = batch
        tasks = {filename: self.encode_image_async(image) 
                for filename, image in zip(filenames, images)}
        
        encoded_images = await asyncio.gather(*tasks.values())

        encode_end = time.time()
        print(f"Encoded {len(images)} images in {encode_end - encode_start:.2f} sec")

        # Return a dictionary mapping filenames to encoded images
        return {filename: enc_img for filename, enc_img in zip(tasks.keys(), encoded_images)}

    async def ask_questions(self, enc_image, queries):
        """Runs multiple queries on a single image asynchronously."""
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, self.model.answer_question, enc_image, q, self.tokenizer) for q in queries]
        return await asyncio.gather(*tasks)

    async def process_batch(self, batch, queries):
        """Processes a batch of images with encoding and queries asynchronously."""
        batch_start = time.time()

        # Encode images asynchronously
        encoded_images = await self.encode_images(batch)

        # Prepare async tasks for querying
        tasks = {filename: self.ask_questions(enc_image, queries) for filename, enc_image in encoded_images.items()}
        
        # Run all queries asynchronously
        results = await asyncio.gather(*tasks.values())

        batch_end = time.time()
        print(f"Processed {len(batch)} images in {batch_end - batch_start:.2f} sec")

        # Map results back to filenames
        return {filename: result for filename, result in zip(tasks.keys(), results)}

    def run_batch(self, batch, queries):
        """Wrapper to run batch inference synchronously."""
        return asyncio.run(self.process_batch(batch, queries))


class ImageLoader:
    def __init__(self, folder_path, target_size=(512, 512), max_workers=4):
        self.folder_path = folder_path
        self.target_size = target_size
        self.max_workers = max_workers
        self.image_data = []  # Store loaded images

        # Auto-load images on initialization
        self.load_images()

    def _load_and_preprocess_image(self, image_path):
        """Loads an image from disk and preprocesses it (resize)."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.target_size, Image.LANCZOS)
                return os.path.basename(image_path), img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def load_images(self):
        """Loads and preprocesses all images in parallel, storing them in self.image_data."""
        image_files = [
            os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._load_and_preprocess_image, image_files)

        self.image_data = [result for result in results if result]  # Store as [(filename, PIL.Image)]
        print(f"Loaded {len(self.image_data)} images from {self.folder_path}")

    def batch_images(self, batch_size=8):
        """Generates batches of images while keeping filenames linked."""
        for i in range(0, len(self.image_data), batch_size):
            batch = self.image_data[i : i + batch_size]
            filenames, images = zip(*batch)  # Separate filenames and images
            yield filenames, images

# Example Usage
image_loader = ImageLoader(folder_path="data/images/test_set", target_size=(512, 512), max_workers=8)

# Process batches directly
moondream = MoondreamProcessor()

# Define the 3 queries per image
queries = [
    "is there a grill in this image? answer yes or no",
    "is there an axe in this image? answer yes or no",
    "is there a chair in this image? answer yes or no"
]

# Process batches
batch_size = 4
for batch in image_loader.batch_images(batch_size):
    results = moondream.run_batch(batch, queries)  # Run batch inference

    # Print results
    for filename, answers in results.items():
        print(f"Results for {filename}: {answers}")

    break