from transformers import(
    MobileViTImageProcessor, 
    MobileViTV2ForImageClassification, 
    AutoModelForCausalLM, 
    AutoTokenizer
    )

from PIL import Image
from openai import OpenAI
import litellm

from dotenv import load_dotenv
import os
import json
import asyncio
import torch

from app.utils import prepare_image
from app.core.response_validation import create_dynamic_schema, ImagePrompts, NoCategoriesSchema

# Models
LLMS = {
    'OPENAI': 'gpt-4o-mini',
    'FIREWORKS_LLAMA': 'fireworks_ai/accounts/fireworks/models/llama-v3p2-11b-vision-instruct',
    'FIREWORKS_QWEN': 'fireworks_ai/accounts/fireworks/models/qwen2-vl-72b-instruct',
}

class MobileViTClassifier:
    def __init__(self):
        self.feature_extractor, self.model = self._load_model_and_processor()
    
    def _load_model_and_processor(self):
        """
        Loads and returns the MobileViTV2 model and feature extractor
        """
        feature_extractor = MobileViTImageProcessor.from_pretrained("shehan97/mobilevitv2-1.0-imagenet1k-256")
        model = MobileViTV2ForImageClassification.from_pretrained("shehan97/mobilevitv2-1.0-imagenet1k-256")
        return feature_extractor, model
    
    def predict(self, image):
        """
        Predicts the ImageNet class for a given PIL Image
        
        Args:
            image (PIL.Image): Input image to classify ### TODO -> change to image_path -> this should be unified over all models. VLMS wont take a PIL image.
            
        Returns:
            dict: Prediction results including class label
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
            
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_class_idx]
        
        return {
            "prediction": predicted_class,
            "model": "mobilevit_v2"
        }
    
    
class OpenAIImageClassifier():
    def __init__(self, model_name:str=LLMS['OPENAI']):
        load_dotenv()
        self._validate_environment()
        self.client = OpenAI()
        self.model_name = model_name

    def _validate_environment(self) -> None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:  # This checks for None or empty string
            raise EnvironmentError("OPENAI_API_KEY is not set or empty in the environment variables")

    def _prepare_message(self, image_path:str, prompt:str) -> list[dict]:
        base64_image = prepare_image(image_path)
        message = [
            {"role": "user", 
            "content": [
                {"type": "text", "text": prompt}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
            }
        ]
        return message
    
    def predict(self, image_path:str, categories:list[str]=None) -> dict:
        if categories:
            schema = create_dynamic_schema(categories)
            prompt = ImagePrompts.DEFAULT_PROMPT
        else:
            schema = NoCategoriesSchema
            prompt = ImagePrompts.NO_CATEGORIES_PROMPT

        message = self._prepare_message(image_path, prompt)
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            response_format=schema
        )
        try:
            response_content = response.choices[0].message.parsed
            response_data = response_content.model_dump()
            schema.model_validate(response_data)  # Validate the response against the schema
        except Exception as e:
            raise ValueError(f"Response validation failed: {e}")
        
        return response_data
    
class VisionLanguageModelClassifier():
    def __init__(self, model_name:str=LLMS['FIREWORKS_QWEN']):
        self.model_name = model_name
        self.system_prompt = ImagePrompts.DEFAULT_PROMPT
    
    @staticmethod
    def clean_llm_output(text):
        # Remove markdown indicators
        text = text.replace('```json', '').replace('```', '')
        
        # Remove newlines and extra spaces
        text = text.replace('\n', '').replace('  ', '')
        
        return json.loads(text)
    
    def _prepare_message(self, image_path:str, prompt:str) -> list[dict]:
        base64_image = prepare_image(image_path)
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
                            "url":f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        return messages
    
    def predict(self, image_path:str, categories:list[str]=None) -> dict:
        if categories:
            prompt = ImagePrompts.get_categorized_prompt(categories)
        else:
            pass
            # Write new prompt

        messages = self._prepare_message(image_path, prompt)

        response = litellm.completion(
            model=self.model_name, 
            messages=messages,
        )

        # return self.clean_llm_output(response.choices[0].message.content)
        return response.choices[0].message.content
    

class MoondreamProcessor:
    def __init__(self, model_id="vikhyatk/moondream2", revision="2024-08-26"):
        """Initialize the model once and load it into memory."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer once
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    async def encode_image_async(self, image):
        """Encodes a single image asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.model.encode_image, image)

    async def encode_images(self, batch):
        """Encodes a batch of images asynchronously."""
        
        # Unpack the batch tuple correctly
        filenames, images = batch
        tasks = {filename: self.encode_image_async(image) 
                for filename, image in zip(filenames, images)}
        
        encoded_images = await asyncio.gather(*tasks.values())

        # Return a dictionary mapping filenames to encoded images
        return {filename: enc_img for filename, enc_img in zip(tasks.keys(), encoded_images)}
    
    def parse_query_result(self, queries, results):
        """Parse query strings and model responses into a structured dictionary."""
        parsed_results = {}
        for query, result in zip(queries, results):
            # Handle both "is there a" and "is there an"
            if "is there a " in query:
                category = query.split("is there a ")[1].split(" in this image")[0]
            elif "is there an " in query:
                category = query.split("is there an ")[1].split(" in this image")[0]
            else:
                category = "unknown"  # Fallback for unexpected query formats

            # Normalize and clean the result
            answer = result.strip().lower()  # Convert to lowercase
            if "yes" in answer:
                parsed_results[category] = "yes"
            elif "no" in answer:
                parsed_results[category] = "no"
            else:
                parsed_results[category] = "unknown"  # Handle unexpected responses
        return parsed_results
    
    async def ask_questions(self, enc_image, queries):
        """Runs multiple queries on a single image asynchronously."""
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, self.model.answer_question, enc_image, q, self.tokenizer)
            for q in queries
        ]
        results = await asyncio.gather(*tasks)
        
        # Parse the queries and results into a structured format
        return self.parse_query_result(queries, results)

    async def process_batch(self, batch, queries, output_file="output.json"):
        """Processes a batch of images with encoding and queries asynchronously."""

        # Encode images asynchronously
        encoded_images = await self.encode_images(batch)

        # Prepare async tasks for querying
        tasks = {
            filename: self.ask_questions(enc_image, queries)
            for filename, enc_image in encoded_images.items()
        }

        # Run all queries asynchronously
        results = await asyncio.gather(*tasks.values())

        # Map results back to filenames
        final_results = {filename: result for filename, result in zip(tasks.keys(), results)}

        # # Save results to a JSON file
        # with open(output_file, "w") as f:
        #     json.dump(final_results, f, indent=4)

        return final_results