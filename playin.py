import logging
import os
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from app.utils.data_tool import get_html_data_as_json, create_db_engine
from app.config import TEMP_IMAGE_DIR

class HTMLImageExtractor:
    def __init__(self):
        self.extracted_images = []

    def process_html(self, html_data_list):
        """
        Process multiple HTML documents and extract image information.
        
        Args:
            html_data_list (list): List of dicts containing:
                - response_text: HTML content
                - base_url: Base URL for the HTML
                - domain_id: Unique identifier for the domain
                
        Returns:
            list: Combined list of image data with domain references
        """
        for html_data in html_data_list:
            html = html_data.get('response_text')[0]
            base_url = html_data.get('base_url')[0]
            domain_id = html_data.get('domain_start_id')

            if not all([html, base_url, domain_id]):
                logging.warning(f"Skipping HTML with missing data. Domain ID: {domain_id}")
                continue

            images = self._extract_images_from_html(html, base_url, domain_id)
            self.extracted_images.extend(images)
            
            logging.info(f"Processed HTML for domain {domain_id}, found {len(images)} images")

        return self.extracted_images

    def _extract_images_from_html(self, html, base_url, domain_id):
        """
        Extract image information from a single HTML document.
        """
        soup = BeautifulSoup(html, 'lxml')
        img_tags = soup.find_all('img')
        img_data = []

        for img in img_tags:
            img_attributes = img.attrs
            img_url = img_attributes.get("src")
            
            if not img_url:
                continue

            if urlparse(img_url).scheme == "":
                img_url = urljoin(base_url, img_url)

            img_url = img_url.replace("\\", "/")

            img_data.append({
                "src": img_url,
                "domain_id": domain_id,
                "original_attributes": img_attributes,
                "base_url": base_url
            })

        return img_data

class ImageDownloader:
    def __init__(self, download_folder=TEMP_IMAGE_DIR):
        self.download_folder = download_folder
        self.default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }
        os.makedirs(download_folder, exist_ok=True)

    def download_bulk_images(self, image_data_list):
        """
        Downloads multiple images and updates their local paths.
        
        Args:
            image_data_list (list): List of image data dictionaries
            
        Returns:
            list: Updated image data list with local paths
        """
        for img_data in image_data_list:
            img_url = img_data.get("src")
            domain_id = img_data.get("domain_id")
            
            if not (img_url and domain_id):
                continue

            if urlparse(img_url).scheme in ["http", "https"]:
                original_name = os.path.basename(urlparse(img_url).path)
                if not original_name:
                    original_name = "unnamed_image.jpg"
                
                # Create filename with domain_id prefix
                img_name = f"{domain_id}_{original_name}"
                img_path = os.path.join(self.download_folder, img_name)
                
                try:
                    response = requests.get(
                        img_url, 
                        headers=self.default_headers, 
                        stream=True, 
                        verify=False
                    )
                    response.raise_for_status()
                    
                    with open(img_path, "wb") as img_file:
                        for chunk in response.iter_content(1024):
                            img_file.write(chunk)
                    
                    logging.info(f"Downloaded image: {img_path}")
                    img_data["local_path"] = img_path
                    
                except Exception as e:
                    logging.warning(f"Failed to download image {img_url} for domain {domain_id}: {e}")
            else:
                logging.warning(f"Skipping invalid URL: {img_url}")
        
        return image_data_list

# Example usage in main:
if __name__ == "__main__":
    assert load_dotenv()
    engine = create_db_engine()
    html_data_list = get_html_data_as_json(engine)

    # Extract images from all HTMLs
    extractor = HTMLImageExtractor()
    all_image_data = extractor.process_html(html_data_list['data'])
    logging.info(f"Total images found: {len(all_image_data)}")

    # Download all images
    downloader = ImageDownloader()
    downloaded_image_data = downloader.download_bulk_images(all_image_data)
    logging.info("Completed bulk image downloads")

    # Optional: Save the results
    output_file = "bulk_image_processing_results.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(downloaded_image_data, file, indent=4)
    logging.info(f"Results saved to {output_file}")