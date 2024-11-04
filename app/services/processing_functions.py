from .image_models import load_model_and_processor_apple_model, predict_image_class_apple_model
from .extract_images import download_images_with_local_path, extract_img_attributes
from collections import defaultdict
from ..config import TEMP_IMAGE_DIR

def process_single_domain(domain_data, model, processor):
    domain_results = {
        "domain_start_id": domain_data["domain_start_id"],
        "predictions": [],
        "statistics": defaultdict(int)  # Will be populated based on actual predictions
    }
    
    # Process images
    for html in domain_data["response_text"]:
        img_data = extract_img_attributes(html)
        # print(img_data)
        # break
        for img in img_data:
            # First download the image and add local path
            download_images_with_local_path([img], TEMP_IMAGE_DIR)
            
            # Then check if download was successful and local path was added
            if "local_path" in img:
                # Skip SVG files since they can't be processed by the model
                if img["local_path"].lower().endswith('.svg'): ### TODO this is a wrong place to check it, we shouldnt download SVGs in the first place
                    continue
                prediction = predict_image_class_apple_model(img["local_path"], model, processor)
                
                # Store individual prediction
                domain_results["predictions"].append({
                    "image_path": img["local_path"],
                    "predicted_class": prediction
                })
                
                # Update statistics counter
                domain_results["statistics"][prediction] += 1
    
    return domain_results

def process_domains(domains_data, output_type="detailed"):
    model, processor = load_model_and_processor_apple_model()
    
    detailed_results = []
    summary_stats = {
        "total_domains": 0,
        "total_images": 0,
        "statistics": defaultdict(int)
    }
    
    for domain in domains_data["data"]:
        domain_results = process_single_domain(domain, model, processor)
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