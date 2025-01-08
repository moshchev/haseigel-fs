# import asyncio
# from app.core.image_models import MoondreamProcessor, ImageLoader
# from PIL import Image

# import time




# if __name__ == "__main__":
#     # Define input directory and batch size
#     image_folder = "data/images/test_set"
#     batch_size = 4
#     output_file = "results.json"

#     # Define queries for the model
#     queries = [
#         "is there a cat in this image? answer yes or no",
#         "is there a dog in this image? answer yes or no",
#         "is there a giraffe in this image? answer yes or no",
#     ]

#     # Initialize the image loader
#     image_loader = ImageLoader(folder_path=image_folder, target_size=(512, 512), max_workers=8)

#     # Initialize the MoondreamProcessor
#     moondream_processor = MoondreamProcessor()

#     # Process images in batches
#     for batch in image_loader.batch_images(batch_size=batch_size):
#         print("Processing a new batch of images...")
#         results = asyncio.run(moondream_processor.process_batch(batch, queries, output_file))

#         # Print results for this batch
#         for filename, answers in results.items():
#             print(f"Results for {filename}: {answers}")

#         # Break after one batch for testing (remove this line for full processing)
#         break

#     print(f"Results saved to {output_file}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import torch

def moondream_transformers():
    start_time = time.time()

    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    # Add device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    ).to(device)  # Move model to GPU
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    model_load_end = time.time()

    print(f"Using device: {device}")  # Print which device is being used
    print(f"Model loaded in {model_load_end - start_time:.2f} seconds")

    # Image processing time
    image_process_start = time.time()

    image = Image.open('data/images/temp/Wintergrillen 992x661.jpg.webp')
    enc_image = model.encode_image(image)
    image_process_end = time.time()
    print(f"Image processed in {image_process_end - image_process_start:.2f} seconds")

    query_start = time.time()
    answer = model.answer_question(enc_image, 'List all object classes detected in the image.', tokenizer)
    query_end = time.time()
    print(f"Query 1 answered in {query_end - query_start:.2f} seconds")
    print(answer)
    return answer 

answer = moondream_transformers()
print(answer)


# import nltk
# from nltk import word_tokenize, pos_tag, RegexpParser
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('punkt_tab')

# # Tokenize and POS tag the text
# tokens = word_tokenize(answer)
# pos_tags = pos_tag(tokens)

# # Define a grammar for noun phrases (NP)
# grammar = r"""
#     NP: {<DT|PP\$>?<JJ>*<NN>+}   # Determiner/possessive, adjectives, and noun(s)
#         {<NNP>+}                 # Proper noun(s)
# """

# # Create a chunk parser and parse the POS-tagged tokens
# chunk_parser = RegexpParser(grammar)
# tree = chunk_parser.parse(pos_tags)

# # Extract noun phrases from the parse tree
# class_list = []
# for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
#     phrase = " ".join(word for word, tag in subtree.leaves())
#     class_list.append(phrase.lower())  # Normalize to lowercase

# # Remove duplicates
# unique_classes = list(set(class_list))

# # Remove articles (a, an, the) from the beginning of phrases
# cleaned_classes = [cls.split(" ", 1)[1] if cls.split(" ", 1)[0] in {"a", "an", "the"} else cls for cls in unique_classes]

# # Print the cleaned classes
# print("Extracted Classes (without articles):", cleaned_classes)