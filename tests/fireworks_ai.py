from litellm import completion
from dotenv import load_dotenv

load_dotenv()

response = completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)

print(response)