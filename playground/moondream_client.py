import moondream as md
from PIL import Image
import time
import psutil

start_time = time.time()

# Initialize model
model = md.vl(model="moondream-2b-int8.mf")
model_load_end = time.time()
print(f"Model loaded in {model_load_end - start_time:.2f} seconds")

# Load and process image
image_start = time.time()
image = Image.open("data/images/temp/Wintergrillen 992x661.jpg.webp")
encoded_image = model.encode_image(image)
image_end = time.time()
print(f"Image processed in {image_end - image_start:.2f} seconds")


# Ask questions
query_start = time.time()
answer1 = model.query(encoded_image, "Is there a grill in this image? Answer in yes or no.")["answer"]
query_end = time.time()
print(f"Query 1 answered in {query_end - query_start:.2f} seconds")
# print("Answer:", answer1)

query_start = time.time()
answer2 = model.query(encoded_image, "Is there a axe in this image? Answer in yes or no.")["answer"]
query_end = time.time()
print(f"Query 2 answered in {query_end - query_start:.2f} seconds")
# print("Answer:", answer2)

query_start = time.time()
answer3 = model.query(encoded_image, "Is there a chair in this image? Answer in yes or no.")["answer"]
query_end = time.time()
print(f"Query 3 answered in {query_end - query_start:.2f} seconds")
# print("Answer:", answer3)

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")