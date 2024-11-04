from dotenv import load_dotenv
from utils.get_htmls_as_df import get_html_data_as_json, create_db_engine
from services.processing_functions import process_domains



if __name__ == "__main__":
    assert load_dotenv()
    engine = create_db_engine()
    input_data = get_html_data_as_json(engine)
    print(process_domains(input_data))