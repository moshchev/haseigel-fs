import litellm
import os

response = litellm.completion(
    model="openai/Qwen/Qwen2-VL-7B-Instruct",               # add `openai/` prefix to model so litellm knows to route to OpenAI                 # api key to your openai compatible endpoint
    api_base="https://xhhoudosagfr3y-8000.proxy.runpod.net/v1",     # set API Base of your Custom OpenAI Endpoint
    messages=[
                {
                    "role": "user",
                    "content": "Hey, how's it going?",
                }
    ],
)
print(response)