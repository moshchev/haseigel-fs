## Small test set on clean images of animals

import os
import requests
from duckduckgo_search import DDGS
from tqdm import tqdm
from urllib.parse import urlparse

# Configuration
num_images = 100  # Total images to download
search_terms = ["cat", "dog", "elephant", "lion", "giraffe"]  # Add more animals if needed
output_folder = "data/images/test_set"  # Change the folder as needed

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def download_image(url, folder):
    """Downloads an image and saves it to the specified folder."""
    try:
        response = requests.get(url, timeout=5, stream=True)
        response.raise_for_status()

        # Extract filename from URL
        filename = os.path.join(folder, os.path.basename(urlparse(url).path))
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
            filename += ".jpg"  # Default extension

        # Save the image
        with open(filename, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        return filename
    except Exception as e:
        return None

def search_and_download_images():
    """Searches and downloads images for given search terms."""
    downloaded = 0
    for term in search_terms:
        print(f"Searching images for: {term}")
        with DDGS() as ddgs:
            results = list(ddgs.images(term, max_results=num_images // len(search_terms)))

        for result in tqdm(results, desc=f"Downloading {term} images"):
            if downloaded >= num_images:
                return
            filename = download_image(result["image"], output_folder)
            if filename:
                downloaded += 1

if __name__ == "__main__":
    search_and_download_images()
    print(f"Downloaded {num_images} images to {output_folder}")
