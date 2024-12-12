from flask import request, jsonify
from app import app  # Import the Flask app instance
from app.services.processing_functions import process_domains, process_html
from app.services.single_image_classification import classify_image
from app.core.image_models import MobileViTClassifier

@app.route('/process-domains', methods=['POST'])
def process_domains_endpoint():
    try:
        data = request.json
        output_type = data.get('output_type', 'detailed')
        
        # Call the processing function
        result = process_domains(
            domains_data=data,
            output_type=output_type
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200


@app.route('/model/<model_name>', methods=['POST'])
def model_classification(model_name):
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Call your classification service
        result = classify_image(image_file, model_name)
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/process-html', methods=['POST'])
def process_html_endpoint():
    try:
        data = request.json
        html = data.get('html')
        if not html:
            return jsonify({'error': 'No HTML content provided'}), 400
        
        model = MobileViTClassifier()
        
        # Call the processing function
        result = process_html(html, model)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
