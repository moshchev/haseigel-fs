from litellm import completion
import litellm
from dotenv import load_dotenv
import os
import sys
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.response_validation import ImagePrompts , NoCategoriesSchema , create_dynamic_schema

load_dotenv()
model_qwen = 'fireworks_ai/accounts/fireworks/models/qwen2-vl-72b-instruct'
img = 'data/images/temp/Wintergrillen 992x661.jpg.webp'


# # encode image to base64
with open(img, 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

schema = create_dynamic_schema(['grill', 'axe', 'hammer'])
prompt = ImagePrompts.NO_CATEGORIES_PROMPT

litellm.enable_json_schema_validation=True

response = completion(
    model=model_qwen, 
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{encoded_string}"
                    }
                }
            ]
        }
    ],
    # response_format=schema
)
print(response)
