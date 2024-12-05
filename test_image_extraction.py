# add logging and other stuff
from dotenv import load_dotenv
from app.utils.data_tool import get_html_data_as_json, create_db_engine, get_random_html
import requests
from collections import defaultdict
from app.config import TEMP_IMAGE_DIR
from app.core.image_models import MobileViTClassifier
import logging
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
import ssl


logname = "application_image_extraction.log"  

logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


def process_html(html, base_url, model):
    """nahui ne nuzhna -> replace with the download and classify

    Args:
        html (str): html code
        model (MobileViTClassifier): keep this model as it is
    Returns:
       (dict) 
        html_results = {
            "predictions": [],
            "statistics": defaultdict(int)
            }
    """
    html_results = {
        "predictions": [],
        "statistics": defaultdict(int)
    }
    
    img_data = extract_img_attributes(html, base_url) # TODO -> replace the download_images_with_local_path -> adjust the workflow
    # instead of putting single images in the function as an input dump whole list there.
    for img in img_data:
        # First download the image and add local path
        download_images_with_local_path([img], TEMP_IMAGE_DIR)
        
        # Then check if download was successful and local path was added
        if "local_path" in img:
            # Skip non jpg files and logo images since they can't be processed by the model, check jpeg!!!
            if not img["local_path"].lower().endswith(".jpg") or "logo" in img["local_path"].lower():
                continue

            ### from here you need to put this in the donwload & classify function
            prediction = model.predict(img["local_path"])['prediction']
            
            # Store individual prediction
            html_results["predictions"].append({
                "image_path": img["local_path"],
                "predicted_class": prediction
            })
            
            # Update statistics counter
            html_results["statistics"][prediction] += 1
    
    return html_results


def extract_img_attributes(html, base_url):
    """
    Parses the HTML to extract attributes of all <img> tags and processes the 'src' attribute.

    Args:
        html (str): The HTML content.
        base_url (str): The base URL to resolve relative paths in 'src' attributes.

    Returns:
        list: A list of dictionaries containing attributes of each <img> tag, with the 'src' modified.
    """
    from urllib.parse import urljoin, urlparse
    from bs4 import BeautifulSoup

    # Parse the HTML content
    soup = BeautifulSoup(html, 'lxml')

    # Find all <img> tags
    img_tags = soup.find_all('img')

    # Initialize list to store each img tag's attributes as dictionaries
    img_data = []

    # Loop through each img tag and extract attributes
    for img in img_tags:
        img_attributes = img.attrs  # Get all attributes of the img tag as a dictionary
        img_url = img_attributes.get("src")  # Get the 'src' attribute
        
        # Convert relative URLs to absolute URLs
        if img_url and urlparse(img_url).scheme == "":
            img_url = urljoin(base_url, img_url)
            logging.info(f"Converted relative URL to absolute: {img_url}")

        # Replace backslashes with forward slashes
        if img_url:
            img_url = img_url.replace("\\", "/")

        # Update the 'src' attribute in the dictionary
        img_attributes["src"] = img_url
        img_data.append(img_attributes)  # Append dictionary to the list

    return img_data

def save_combined_html(df, output_file="../data/combined.html"):
    # Combine all response text into one HTML file
    with open(output_file, "w", encoding="utf-8") as file:
        for html_content in df["response_text"]:
            file.write(html_content)
            file.write("\n")  # Separate each HTML content by a newline for readability

def download_images_with_local_path(dict_list, download_folder="../data/images"):
    """
    Downloads images from URLs in a list of dictionaries and adds local file paths.
    
    Args:
        dict_list (list): List of dictionaries containing image attributes.
            Each dictionary has keys like 'src', 'alt', 'title', etc.
            Example:
            [
                {
                    'title': 'some title',
                    'alt': 'some alt text',
                    'border': '0', 
                    'src': 'https://example.com/images/logo.png' - URL to the image
                },
                ...
            ]
        download_folder (str): Path where images will be downloaded to.
            Defaults to "../data/images"
            
    Returns:
        None: Modifies the input dictionaries in-place by adding 'local_path' key
    """

    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)
    default_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
    
    # Loop through each dictionary in the list
    for index, img_data in enumerate(dict_list):
        img_url = img_data.get("src")
        
        # Check if URL is valid and not a data URI
        if img_url and urlparse(img_url).scheme in ["http", "https"]:
            # Generate filename consistent with the original function
            img_name = f"{index}_{os.path.basename(urlparse(img_url).path)}"
            img_path = os.path.join(download_folder, img_name)
            
            # Download the image using requests with SSL verification disabled
            try:
                # Attempt to download with custom User-Agent on the first try
                response = requests.get(img_url, headers=default_headers, stream=True, verify=False)
                response.raise_for_status()  # Raise an error for HTTP issues
                with open(img_path, "wb") as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
                logging.info(f"Downloaded: {img_name}")
                
                # Add the relative path to the dictionary
                img_data["local_path"] = os.path.relpath(img_path, start=download_folder) #TODO -> returns the name of the local file instead of the path to the file 
                
            except Exception as e:
                # Print the error and skip to the next item
                logging.warning(f"Error downloading {img_url}: {e}")
        else:
            # Skip data URIs or invalid URLs
            if img_url and img_url.startswith("data:image"):
                logging.warning(f"Skipping item {index} - Data URI found.")
            else:
                logging.warning(f"Skipping item {index} - No valid image URL found.")

if __name__ == "__main__":
    assert load_dotenv()
    engine = create_db_engine()
    # input_data = get_html_data_as_json(engine)
    input_data = get_random_html(engine)
    logging.info('got the data')
    # Extract HTML content and base URL from input_data
    html = input_data.get("response_text", "")
    base_url = input_data.get("base_url", "")

    # Validate input data
    if not html:
        logging.error("No HTML content found in input data.")
        exit(1)
    if not base_url:
        logging.error("No base URL found in input data.")
        exit(1)

    # Initialize the MobileViTClassifier
    model = MobileViTClassifier()

    logging.info("Processing HTML content.")
    results = process_html(html, base_url, model)

    # Output results
    print("Processing Results:")
    print(results)

    # Save results to a file
    output_file = "image_processing_results.json"
    with open(output_file, "w", encoding="utf-8") as file:
        import json
        json.dump(results, file, indent=4)
    logging.info(f"Results saved to {output_file}.")
