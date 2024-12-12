from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve
import requests
from app.config import TEMP_IMAGE_DIR

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
            print(f"Converted relative URL to absolute: {img_url}")

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

def download_images_with_local_path(dict_list, download_folder=TEMP_IMAGE_DIR):
    """
    Downloads images from URLs in a list of dictionaries and adds local file paths.
    
    Args:
        dict_list (list): List of dictionaries containing image attributes.
        download_folder (str): Path where images will be downloaded to.
            Defaults to the TEMP_IMAGE_DIR from config.
    """
    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)
    default_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
    
    for index, img_data in enumerate(dict_list):
        img_url = img_data.get("src")
        
        # Check if URL is valid and not a data URI
        if img_url and urlparse(img_url).scheme in ["http", "https"]:
            # Generate a filename for the image
            img_name = f"{index}_{os.path.basename(urlparse(img_url).path)}"
            img_path = os.path.join(download_folder, img_name)  # Full path to the image
            
            try:
                # Download the image
                response = requests.get(img_url, headers=default_headers, stream=True, verify=False)
                response.raise_for_status()
                with open(img_path, "wb") as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
                print(f"Downloaded image: {img_path}")
                
                # Set the absolute path in the dictionary
                img_data["local_path"] = img_path
            except Exception as e:
                print(f"Failed to download image {img_url}: {e}")
        else:
            print(f"Skipping invalid URL: {img_url}")

# if __name__ == "__main__":
#     # Load the parquet file
#     df = pd.read_parquet('haseigel-fs/data/HTML_data.parquet')

#     # Save all HTML content into one combined HTML file
#     save_combined_html(df)

#     # Read combined HTML for parsing
#     with open("haseigel-fs/data/combined.html", "r", encoding="utf-8") as file:
#         combined_html = file.read()

#     # Extract image attributes into a dictionary list
#     img_data = extract_img_attributes(combined_html)

#     # Download images and add local paths to the dictionary list
#     download_images_with_local_path(img_data)

#     # Optionally, convert img_data to a DataFrame and save
#     img_df = pd.DataFrame(img_data)
#     img_df.to_csv("data/image_attributes_with_local_path.csv", index=False)  # Save to CSV for further analysis