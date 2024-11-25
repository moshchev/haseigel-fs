import os
import sys
# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from dotenv import load_dotenv
from app.utils.data_tool import get_html_data_as_json, create_db_engine, get_random_html
import requests

def test_domains():
    assert load_dotenv()
    engine = create_db_engine()
    input_data = get_html_data_as_json(engine)
    # input_data = get_random_html(engine)
    print('got the data')

    # Send POST request to localhost with input data
    response = requests.post('http://127.0.0.1:5000/process-domains', json=input_data)
    # response = requests.post('http://127.0.0.1:5000/process-html', json=input_data)
    # Check if request was successful
    if response.status_code == 200:
        print('Successfully sent data to server')
        print(response.json())
    else:
        print(f'Error sending data: {response.status_code}')
        print(response.text)

def test_image():
    image_path = '/Users/alexander/Desktop/projects/haseigel-fs/data/images/temp/0_1156A.jpeg-200x200.jpg'
    response = requests.post('http://127.0.0.1:5000/model/mobilevit_v2', files={'image': open(image_path, 'rb')})
    print(response.json())

if __name__ == "__main__":
    test_image()