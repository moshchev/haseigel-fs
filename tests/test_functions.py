import os
import sys
# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.utils.data_tool import get_html_data_as_json, create_db_engine
from dotenv import load_dotenv
from app.services.processing_functions import process_domains

def test_html_data():
    assert load_dotenv()
    engine = create_db_engine()
    input_data = get_html_data_as_json(engine)
    return input_data

if __name__ == "__main__":
    inputs = test_html_data()
    outputs = process_domains(inputs)
    print(outputs)