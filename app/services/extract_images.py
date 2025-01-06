from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve
import requests
from app.config import TEMP_IMAGE_DIR
from urllib.parse import urljoin, urlparse

def extract_img_attributes(html, base_url):
    """
    Parses the HTML to extract attributes of all <img> tags and processes the 'src' attribute.
    Filters out duplicate image URLs.

    Args:
        html (str): The HTML content.
        base_url (str): The base URL to resolve relative paths in 'src' attributes.

    Returns:
        list: A list of dictionaries containing unique attributes of each <img> tag.
    """

    # Parse the HTML content
    soup = BeautifulSoup(html, 'lxml')

    # Find all <img> tags
    img_tags = soup.find_all('img')

    # Initialize list to store each img tag's attributes as dictionaries
    img_data = []
    seen_urls = set()  # Keep track of URLs we've already processed

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
            
            # Only add the image if we haven't seen this URL before
            if img_url not in seen_urls:
                seen_urls.add(img_url)
                img_attributes["src"] = img_url
                img_data.append(img_attributes)
    
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
    Includes domain_id in the filename.
    """
    os.makedirs(download_folder, exist_ok=True)
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    
    for img_data in dict_list:
        img_url = img_data.get("src")
        domain_id = img_data.get("domain_id")
        
        if img_url and urlparse(img_url).scheme in ["http", "https"]:
            parsed_url = urlparse(img_url)
            original_name = os.path.basename(parsed_url.path)
            img_name = f"{domain_id}_{original_name}"
            img_path = os.path.join(download_folder, img_name)
            
            try:
                # Try with verification first
                response = requests.get(
                    img_url, 
                    headers=default_headers, 
                    stream=True, 
                    verify=True  # Enable certificate verification
                )
                response.raise_for_status()
                
                with open(img_path, "wb") as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
                print(f"Downloaded image: {img_path}")
                img_data["local_path"] = img_path
                
            except requests.exceptions.SSLError:
                # If SSL verification fails, try without verification
                print(f"SSL verification failed for {img_url}, retrying without verification...")
                try:
                    response = requests.get(
                        img_url, 
                        headers=default_headers, 
                        stream=True, 
                        verify=False
                    )
                    response.raise_for_status()
                    
                    with open(img_path, "wb") as img_file:
                        for chunk in response.iter_content(1024):
                            img_file.write(chunk)
                    print(f"Downloaded image (insecure): {img_path}")
                    img_data["local_path"] = img_path
                    
                except Exception as e:
                    print(f"Failed to download image {img_url}: {e}")
                    
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