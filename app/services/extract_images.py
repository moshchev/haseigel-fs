from bs4 import BeautifulSoup
from urllib.parse import urlparse
from urllib.parse import urljoin, urlparse
import os
import requests
import logging
from app.config import TEMP_IMAGE_DIR

from typing import List, Dict, Any

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


def download_images_with_local_path(dict_list, download_folder=TEMP_IMAGE_DIR):
    """
    Downloads images from URLs in a list of dictionaries and adds local file paths.
    Includes domain_id in the filename.
    """
    os.makedirs(download_folder, exist_ok=True)
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    
    # Define timeouts
    TIMEOUT = (5, 15)  # (connect timeout, read timeout)
    
    for img_data in dict_list:
        img_url = img_data.get("src")
        domain_id = img_data.get("domain_id")
        
        if not img_url or urlparse(img_url).scheme not in ["http", "https"]:
            print(f"Skipping invalid URL: {img_url}")
            continue
            
        parsed_url = urlparse(img_url)
        original_name = os.path.basename(parsed_url.path)
        
        # Skip if filename is empty
        if not original_name:
            print(f"Skipping URL with no filename: {img_url}")
            continue
            
        img_name = f"{domain_id}_{original_name}"
        img_path = os.path.join(download_folder, img_name)
        
        try:
            # Try with verification first
            response = requests.get(
                img_url, 
                headers=default_headers, 
                stream=True, 
                verify=True,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            
            # Check if content type is image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                print(f"Skipping non-image content type ({content_type}): {img_url}")
                continue
                
            # Check file size before downloading
            content_length = int(response.headers.get('content-length', 0))
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                print(f"Skipping large image ({content_length/1024/1024:.2f}MB): {img_url}")
                continue
            
            with open(img_path, "wb") as img_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        img_file.write(chunk)
            print(f"Downloaded image: {img_path}")
            img_data["local_path"] = img_path
            
        except requests.exceptions.SSLError:
            print(f"SSL verification failed for {img_url}, retrying without verification...")
            try:
                response = requests.get(
                    img_url, 
                    headers=default_headers, 
                    stream=True, 
                    verify=False,
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                
                with open(img_path, "wb") as img_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            img_file.write(chunk)
                print(f"Downloaded image (insecure): {img_path}")
                img_data["local_path"] = img_path
                
            except requests.exceptions.Timeout:
                print(f"Timeout downloading image {img_url}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download image {img_url}: {str(e)}")
                
        except requests.exceptions.Timeout:
            print(f"Timeout downloading image {img_url}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download image {img_url}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error downloading {img_url}: {str(e)}")


def download_images(image_data: List[Dict[str, List[str]]], temp_dir: str) -> List[Dict[str, Any]]:
    """
    Downloads all images from the collected image data.
    
    Args:
        image_data: List of dictionaries with structure {'domain_id': id, 'images': [urls]}
        temp_dir: Directory to store downloaded images
        
    Returns:
        List of dictionaries containing downloaded image information
    """
    downloaded_images = []
    
    for domain_data in image_data:
        domain_id = domain_data['domain_id']
        image_urls = domain_data['images']
        
        # Create list of dicts with URLs and domain_id
        images_to_download = [
            {'src': url, 'domain_id': domain_id}
            for url in image_urls
        ]
        
        # Download images with domain-specific names
        download_images_with_local_path(images_to_download, temp_dir)
        
        # Filter out failed downloads
        downloaded_images.extend([img for img in images_to_download if img.get("local_path")])
    
    return downloaded_images