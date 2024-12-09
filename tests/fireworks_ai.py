from litellm import completion
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.response_validation import ImagePrompts
from app.utils.image_preprocessing import encode_image_to_base64

load_dotenv()
model="fireworks_ai/accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
# img = '/Users/alexander/Desktop/projects/haseigel-fs/app/data/images/temp/0_cuscinetti_a_sfere-315x242.jpg'
img = '/Users/alexander/Desktop/projects/haseigel-fs/data/images/temp/Wintergrillen 992x661.jpg.webp'
base64_img = encode_image_to_base64(img)

response = completion(
    model=model, 
    messages=[
       {"role": "user", "content": ImagePrompts.get_categorized_prompt(['grill', 'axe', 'hammer'])},
       {"role": "user", "content": f"data:image/jpeg;base64,{base64_img}"}
   ],
)
print(response.choices[0].message.content)