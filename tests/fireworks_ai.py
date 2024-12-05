from litellm import completion
from dotenv import load_dotenv

load_dotenv()

# response = completion(
#     model="fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct", 
#     messages=[
#        {"role": "user", "content": "hello from litellm"}
#    ],
# )

# print(response)

import requests
import json
import os

url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
  "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
  "max_tokens": 4096,
  "top_p": 1,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.6,
  "messages": [
    {"role": "user", "content": "hello from frankfurt"}
  ]
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {os.getenv('FIREWORKS_AI_API_KEY')}"
}
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

print(response.text)