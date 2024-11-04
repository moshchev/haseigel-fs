from dotenv import load_dotenv
from app.utils.get_htmls_as_df import get_html_data_as_json, create_db_engine
import requests

if __name__ == "__main__":
    assert load_dotenv()
    engine = create_db_engine()
    input_data = get_html_data_as_json(engine)
    print('got the data')

    # Send POST request to localhost with input data
    response = requests.post('http://127.0.0.1:5000/process-domains', json=input_data)

    # Check if request was successful
    if response.status_code == 200:
        print('Successfully sent data to server')
        print(response.json())
    else:
        print(f'Error sending data: {response.status_code}')
        print(response.text)

    # response = requests.get('http://127.0.0.1:5000/health')
    # print("Status Code:", response.status_code)
    # print("Response Text:", response.text)
    # print("Response Headers:", response.headers)