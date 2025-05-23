from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

def create_db_engine():
    # Retrieve database credentials from environment variables
    host = os.getenv('DB_HOST')
    database = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    port = os.getenv('DB_PORT')

    # Construct the PostgreSQL connection string
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    return create_engine(connection_string)

# SQL query to fetch data
def load_and_save_html_data(engine, limit=250000):
    # SQL query to fetch data
    query = f"SELECT * FROM html_data LIMIT {limit}"

    # Execute query and load data into DataFrame
    df = pd.read_sql(query, engine)

    # Define the relative path to the data folder
    data_path = 'data'

    # Ensure the 'data' directory exists
    os.makedirs(data_path, exist_ok=True)

    # Save the DataFrame to a parquet file in the specified data folder
    df.to_parquet(os.path.join(data_path, 'HTML_data.parquet'))
    print(f"Data saved to '{data_path}/HTML_data.parquet'")
    
    return df


def get_html_data_as_json(engine, limit=15):
    """
    Fetches HTML data from database and restructures it into a JSON-like format
    where response_text is grouped by domain_start_id.

    Args:
        engine: SQLAlchemy engine instance
        limit (int): Maximum number of rows to fetch

    Returns:
        dict: Data structured as:
        {
            "data": [
                {
                    "domain_start_id": id,
                    "response_text": [html1, html2, ...]
                },
                ...
            ]
        }
    """
    # SQL query to fetch data
    query = f"SELECT domain_start_id, response_url, response_text FROM html_data LIMIT {limit}"
    
    # Execute query and load data into DataFrame
    df = pd.read_sql(query, engine)
    
    # Group by domain_start_id and aggregate response_text into lists
    grouped = df.groupby('domain_start_id')[['response_url', 'response_text']].agg(list).reset_index()
    
    # Convert to desired format
    result = {
        "data": [
            {
                "domain_start_id": row['domain_start_id'],
                "base_url": row['response_url'],
                "response_text": row['response_text']
            }
            for _, row in grouped.iterrows()
        ],
        "categories": input("Enter categories or click enter to skip: ")
    }
    
    return result

def get_random_html(engine):
    """
    Fetches a single random HTML data from the database and returns it as a dictionary.

    Args:
        engine: SQLAlchemy engine instance

    Returns:
        dict: Data structured as:
        {
            "html": actual_html_content
        }
    """
    # SQL query to fetch a single random row
    # query = "SELECT response_text FROM html_data ORDER BY RANDOM() LIMIT 1"
    query = 'SELECT * FROM html_data LIMIT 1' # debug
    
    # Execute query and load data into DataFrame
    df = pd.read_sql(query, engine)
    
    # Extract the HTML content
    if not df.empty:
        html_content = df.iloc[0]['response_text']
        base_url = df.iloc[0]['response_url']
        return {
            "response_text": html_content,
            "base_url": base_url
        }
    else:
        return {"html": None, "base_url": None}
