from flask import request, jsonify, Blueprint
from app.services.processing_functions import process_domains, process_html
from app.services.single_image_classification import classify_image
from app.core.image_models import MobileViTClassifier
from app.config import ERROR_MESSAGES, DEFAULT_OUTPUT_TYPE
from app.services.process_domains_moondream import process_domains_moondream_service

# Create blueprint
api = Blueprint('api', __name__)


@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200


@api.route('/model/<model_name>', methods=['POST'])
def model_classification(model_name):
    try:
        if 'image' not in request.files:
            return jsonify({'error': ERROR_MESSAGES['NO_IMAGE']}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': ERROR_MESSAGES['NO_FILE_SELECTED']}), 400
            
        result = classify_image(image_file, model_name)
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/process-domains', methods=['POST'])
def process_domains_endpoint():
    try:
        data = request.json
        output_type = data.get('output_type', DEFAULT_OUTPUT_TYPE)
        
        result = process_domains(
            domains_data=data,
            output_type=output_type
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@api.route('/process-html', methods=['POST'])
def process_html_endpoint():
    """
    {'html':'actual html'}
    """
    try:
        data = request.json
        html = data.get('response_text')
        base_url = data.get('response_url')
        if not html:
            return jsonify({'error': ERROR_MESSAGES['NO_HTML_CONTENT']}), 400
        
        model = MobileViTClassifier()
        
        result = process_html(html, base_url, model)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

@api.route('/process-domains-moondream', methods=['POST'])
def process_domains_moondream_endpoint():
    try:
        data = request.json
        categories = data.get('categories')
        result = process_domains_moondream_service(data, categories)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
