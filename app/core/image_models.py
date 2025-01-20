from transformers import(
    MobileViTImageProcessor, 
    MobileViTV2ForImageClassification, 
    AutoModelForCausalLM, 
    AutoTokenizer
)

from PIL import Image
from openai import OpenAI
import litellm
import torch
from dotenv import load_dotenv

import os
import json
import asyncio

from app.utils import prepare_image
from app.core.response_validation import (
    create_dynamic_schema, 
    ImagePrompts, 
    NoCategoriesSchema
)

# Models that you can plug into litellm and directly use in the codebase
# you can also add your own models here, they should be compatible with openai api standards
# I recommend using the sglang for serving your own models, and it will be compatible with litellm
# https://docs.litellm.ai/docs/providers/openai_compatible

LLMS = {
    'OPENAI': 'openai/gpt-4o-mini',
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

    async def process_batch(self, batch, queries):
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

        return final_results

class AsyncVisionLanguageModelClassifier():
    def __init__(self, model_name: str = LLMS['FIREWORKS_QWEN']):
        self.model_name = model_name
        self.system_prompt = ImagePrompts.DEFAULT_PROMPT

    @staticmethod
    def clean_llm_output(text):
        text = text.replace('```json', '').replace('```', '')
        text = text.replace('\n', '').replace('  ', '')
        return json.loads(text)

    async def _prepare_message(self, image_path:str, prompt:str) -> list[dict]:
        base64_image = prepare_image(image_path)
        messages = [
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
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        return messages

    async def prepare_batch_messages(
        self,
        image_paths: list[str], 
        categories: list[str] = None, 
        batch_size: int = 2
    ) -> list[list[dict]]:
        """
        Prepare messages for a batch of images asynchronously.
        """
        if categories:
            prompt = ImagePrompts.get_categorized_prompt(categories)
        else:
            prompt = ImagePrompts.NO_CATEGORIES_PROMPT

        all_messages = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            tasks = [self._prepare_message(path, prompt) for path in batch]
            batch_messages = await asyncio.gather(*tasks)
            all_messages.extend(batch_messages)
        
        return all_messages
    
    async def predict(self, image_path:str, categories:list[str]=None) -> dict:
        if categories:
            prompt = ImagePrompts.get_categorized_prompt(categories)
        else:
            prompt = "<INSERT YOUR DEFAULT PROMPT HERE>"

        messages = await self._prepare_message(image_path, prompt)

        response = await litellm.acompletion(
            model=self.model_name, 
            messages=messages,
        )
        return response.choices[0].message.content
    
    async def predict_batch(
        self,
        image_paths: list[str],
        categories: list[str] = None,
        prep_batch_size: int = 20,
        request_batch_size: int = 2
    ) -> list[dict]:
        """
        Processes images in *two* stages:
        1) Prepares all messages in batches (to avoid memory blowup).
        2) Sends those messages to the server in smaller chunks (request_batch_size).
        """
        # Stage 1: Prepare all messages
        batch_messages = await self.prepare_batch_messages(
            image_paths, 
            categories, 
            batch_size=prep_batch_size
        )
        # batch_messages now has one entry per image

        results = []
        # Stage 2: Send messages in smaller chunks, so you don't overload the server
        for i in range(0, len(batch_messages), request_batch_size):
            sub_batch = batch_messages[i:i + request_batch_size]
            sub_tasks = [
                litellm.acompletion(model=self.model_name, messages=messages)
                for messages in sub_batch
            ]
            responses = await asyncio.gather(*sub_tasks)

            # Match each response with its corresponding image path
            for idx, response in enumerate(responses):
                file_path = image_paths[i + idx]
                prediction = response.choices[0].message.content
                results.append({"file_path": file_path, "prediction": prediction})
        
        return results