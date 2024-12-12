from ..core.image_models import MobileViTClassifier
from .extract_images import download_images_with_local_path, extract_img_attributes
from collections import defaultdict
from app.config import TEMP_IMAGE_DIR


def process_html(html, base_url, model):
    """
    Processes HTML to extract image attributes, download images, and classify them.

    Args:
        html (str): HTML content.
        base_url (str): Base URL for resolving relative image paths.
        model (MobileViTClassifier): Classification model.

    Returns:
        dict: Contains predictions and statistics.
    """
    html_results = {
        "predictions": [],
        "statistics": defaultdict(int)
    }

    # Extract image attributes
    img_data = extract_img_attributes(html, base_url)

    # Download images and update local paths
    download_images_with_local_path(img_data, TEMP_IMAGE_DIR)

    # Classify downloaded images
    for img in img_data:
        if "local_path" in img and img["local_path"]:
            if not any(img["local_path"].lower().endswith(ext) for ext in [".jpeg", ".jpg", ".png"]) and "logo" not in img["local_path"].lower():
                continue

            try:
                prediction = model.predict(img["local_path"])['prediction']
                html_results["predictions"].append({
                    "image_path": img["local_path"],
                    "predicted_class": prediction
                })
                html_results["statistics"][prediction] += 1
            except Exception as e:
                print(f"Error classifying image {img['local_path']}: {e}")
    
    return html_results


def process_single_domain(domain_data, model):
    domain_results = {
        "domain_start_id": domain_data["domain_start_id"],
        "predictions": [],
        "statistics": defaultdict(int)
    }
    
    # Process each HTML
    for html in domain_data["response_text"]:
        html_results = process_html(html, model)
        
        # Append predictions and update statistics
        domain_results["predictions"].extend(html_results["predictions"])
        for category, count in html_results["statistics"].items():
            domain_results["statistics"][category] += count
    
    return domain_results


def process_domains(domains_data, output_type="detailed"):
    model = MobileViTClassifier()

    detailed_results = []
    summary_stats = {
        "total_domains": 0,
        "total_images": 0,
        "statistics": defaultdict(int)
    }
    
    for domain in domains_data["data"]:
        domain_results = process_single_domain(domain, model)
        detailed_results.append({
            "domain_start_id": domain["domain_start_id"],
            "statistics": dict(domain_results["statistics"]),  # Convert defaultdict to regular dict
            "predictions": domain_results["predictions"],
            "total_images": len(domain_results["predictions"])
        })
        
        # Update summary
        summary_stats["total_domains"] += 1
        summary_stats["total_images"] += len(domain_results["predictions"])
        for category, count in domain_results["statistics"].items():
            summary_stats["statistics"][category] += count

    # Convert summary_stats['statistics'] back to regular dict
    summary_stats["statistics"] = dict(summary_stats["statistics"])
    
    if output_type == "detailed":
        return {
            "status": "success",
            "output": {
                "details": detailed_results,
                "summary": dict(summary_stats)  # Convert defaultdict to regular dict
            }
        }
    else:
        return {
            "status": "success",
            "output": dict(summary_stats)  # Convert defaultdict to regular dict
        }