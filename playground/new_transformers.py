from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time 

image = Image.open('/Users/christiannikolov/Downloads/HASE&IGEL/haseigel-fs/data/images/temp/Wintergrillen 992x661.jpg.webp')

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    # Uncomment to run on GPU.
    device_map={"": "mps"}
)

# Captioning
start = time.time()
print("Short caption:")
print(model.caption(image, length="short")["caption"])
print(f"Time: {time.time() - start:.2f}s\n")

start = time.time()
print("Normal caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    # Streaming generation example, supported for caption() and detect()
    print(t, end="", flush=True)
print(model.caption(image, length="normal"))
print(f"Time: {time.time() - start:.2f}s\n")

# Visual Querying
start = time.time()
print("Visual query: 'How many people are in the image?'")
print(model.query(image, "How many people are in the image?")["answer"])
print(f"Time: {time.time() - start:.2f}s\n")

# Object Detection
start = time.time()
print("Object detection: 'face'")
objects = model.detect(image, "face")["objects"]
print(f"Found {len(objects)} face(s)")
print(f"Time: {time.time() - start:.2f}s\n")

# Pointing
start = time.time()
print("Pointing: 'person'")
points = model.point(image, "person")["points"]
print(f"Found {len(points)} person(s)")
print(f"Time: {time.time() - start:.2f}s")



