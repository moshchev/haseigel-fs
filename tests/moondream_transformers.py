from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import torch

start_time = time.time()

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
# Add device detection
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
).to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

model_load_end = time.time()

print(f"Using device: {device}")  # Print which device is being used
print(f"Model loaded in {model_load_end - start_time:.2f} seconds")

# Image processing time
image_process_start = time.time()

image = Image.open('data/images/temp/Wintergrillen 992x661.jpg.webp')
enc_image = model.encode_image(image)
image_process_end = time.time()
print(f"Image processed in {image_process_end - image_process_start:.2f} seconds")

query_start = time.time()
answer = model.answer_question(enc_image, "is there a grill in this image? answer yes or no", tokenizer)
query_end = time.time()
print(f"Query 1 answered in {query_end - query_start:.2f} seconds")

query_start = time.time()
answer = model.answer_question(enc_image, "is there a axe in this image? answer yes or no", tokenizer)
query_end = time.time()
print(f"Query 2 answered in {query_end - query_start:.2f} seconds")

query_start = time.time()
answer = model.answer_question(enc_image, "is there a chair in this image? answer yes or no", tokenizer)
query_end = time.time()
print(f"Query 3 answered in {query_end - query_start:.2f} seconds")
# print(answer)


end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")