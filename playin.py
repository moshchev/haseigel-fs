from collections import defaultdict
from typing import List, Dict, Any
from app.services.extract_images import extract_img_attributes, download_images_with_local_path
from app.utils.data_tool import get_html_data_as_json, create_db_engine
from dotenv import load_dotenv
from app.core.image_models import MobileViTClassifier
from urllib.parse import urlparse
from app.config import TEMP_IMAGE_DIR


def collect_image_data(html_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collects image data from multiple HTML and organizes it by domain ID.
    Filters images to include only .jpeg, .jpg, .png extensions and excludes 'logo' in filename.
    
    Args:
        html_data_list: List of dictionaries containing:
            - domain_start_id: ID of the domain
            - base_url: Base URL for the domain
            - response_text: HTML content
        
    Returns:
        List of dictionaries with structure: 
        [{'domain_id': domain_id, 'images': ['http://...', 'http://...']}]
    """
    domain_images = {}
    valid_extensions = ('.jpeg', '.jpg', '.png')
    
    for data in html_data_list:
        domain_id = data['domain_start_id']
        base_url = data['base_url'][0]
        html = data['response_text'][0]
        
        # Get image data for this HTML
        img_data = extract_img_attributes(html, base_url)
        
        # Extract URLs and filter by extension and exclude logos
        image_urls = [
            img.get('src') for img in img_data 
            if img.get('src') and (
                any(img.get('src').lower().endswith(ext) for ext in valid_extensions) and
                'logo' not in img.get('src').lower()
            )
        ]
        
        # Add to domain_images dict
        if domain_id in domain_images:
            domain_images[domain_id].extend(image_urls)
            # Remove duplicates while preserving order
            domain_images[domain_id] = list(dict.fromkeys(domain_images[domain_id]))
        else:
            domain_images[domain_id] = image_urls
    
    # Convert to final format
    result = [
        {'domain_id': domain_id, 'images': images}
        for domain_id, images in domain_images.items()
    ]
    
    return result


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

def process_images_with_model(image_data: List[Dict[str, Any]], model) -> Dict[str, Any]:
    """
    Processes downloaded images with the classification model.
    
    Args:
        image_data: List of dictionaries containing image data with local paths
        model: Classification model instance
        
    Returns:
        Dictionary containing predictions and statistics
    """
    results = {
        "predictions": [],
        "statistics": defaultdict(int)
    }

    for img in image_data:
        if not img.get("local_path"):
            continue

        if not any(img["local_path"].lower().endswith(ext) for ext in [".jpeg", ".jpg", ".png"]) and "logo" not in img["local_path"].lower():
            continue

        try:
            prediction = model.predict(img["local_path"])['prediction']
            results["predictions"].append({
                "image_path": img["local_path"],
                "predicted_class": prediction,
                "original_url": img.get("src", "")
            })
            results["statistics"][prediction] += 1
        except Exception as e:
            print(f"Error classifying image {img['local_path']}: {e}")

    return results

if __name__ == "__main__":
    assert load_dotenv()
    engine = create_db_engine()
    html_data_list = get_html_data_as_json(engine)
    model = MobileViTClassifier()
    
    # Collect image data
    image_data = collect_image_data(html_data_list['data'])
    downloaded_images = download_images(image_data, TEMP_IMAGE_DIR)
    print(downloaded_images)