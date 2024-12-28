from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time

start_time = time.time()

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"  # Pin to specific version
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

model_load_end = time.time()
print(f"Model loaded in {model_load_end - start_time:.2f} seconds")

# Image processing time
image_process_start = time.time()

image = Image.open('data/images/temp/Wintergrillen 992x661.jpg.webp')
enc_image = model.encode_image(image)
print(model.answer_question(enc_image, "is there a grill in this image? answer yes or no", tokenizer))

image_process_end = time.time()
print(f"Image processed in {image_process_end - image_process_start:.2f} seconds")

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
