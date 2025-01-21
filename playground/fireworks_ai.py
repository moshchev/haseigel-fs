import os
import sys
import base64
import getpass
import litellm
from litellm import completion
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.response_validation import ImagePrompts , NoCategoriesSchema , create_dynamic_schema

def get_response(prompt, schema, model):
    litellm.enable_json_schema_validation=True
    try:
        response = completion(
            model=models["llama-v3-8b"], 
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
            response_format=schema
        )
        response_msg = response.choices[0].model_extra['message'].content
        # print(response)
        return response_msg
    except Exception as e:
        return e

if __name__ == "__main__":
    load_dotenv()
    # model, check others from https://fireworks.ai/models
    models = {
        'qwen2-vl-72b-instruct': 'fireworks_ai/accounts/fireworks/models/qwen2-vl-72b-instruct',
        'qwen2-vl-7b-instruct': 'fireworks_ai/accounts/fireworks/models/qwen2-vl-7b-instruct',
        'deepseek-v3': 'fireworks_ai/accounts/fireworks/models/deepseek-v3',
        'llama-v3-8b': 'fireworks_ai/accounts/fireworks/models/llama-v3-8b-instruct',
        'openchat-3p5-0106-7b': 'fireworks_ai/accounts/fireworks/models/openchat-3p5-0106-7b',
        'codegemma-2b': 'fireworks_ai/accounts/fireworks/models/codegemma-2b',
        'StarCoder2 3B': 'accounts/fireworks/models/starcoder2-3b',
        'Llama 3.2 3B': 'accounts/fireworks/models/llama-v3p2-3b',
        'llama 3.2 1B instruct': 'accounts/fireworks/models/llama-v3p2-1b-instruct'
    }

    img = 'data/images/temp/Wintergrillen 992x661.jpg.webp'

    # generate your own api keys from https://fireworks.ai/account/api-keys
    os.environ["FIREWORKS_API_KEY"] = "your_api_key"

    if "FIREWORKS_API_KEY" not in os.environ:
        os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Fireworks API Key:")

    # # encode image to base64
    with open(img, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    schema = create_dynamic_schema(['human', 'grill'])
    prompt = ImagePrompts.NO_CATEGORIES_PROMPT

    for key in models.keys():
        response = get_response(prompt, schema, models[key])
        print(f"Model: {key}")
        print(f"Response: {response}")