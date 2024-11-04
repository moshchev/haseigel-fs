from flask import request, jsonify
from app import app  # Import the Flask app instance
from app.services.processing_functions import process_domains

@app.route('/process-domains', methods=['POST'])
def process_domains_endpoint():
    try:
        data = request.json
        output_type = data.get('output_type', 'detailed')
        
        # Call the processing function
        result = process_domains(
            domains_data=data.get('domains_data'),  # None will fetch from DB
            output_type=output_type
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200