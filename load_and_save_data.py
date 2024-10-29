from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
assert load_dotenv(), "Failed to load .env file"

# Retrieve database credentials from environment variables
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASS')
port = os.getenv('DB_PORT')

# Construct the PostgreSQL connection string
connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
engine = create_engine(connection_string)

# SQL query to fetch data
query = "SELECT * FROM html_data LIMIT 1000"

# Execute query and load data into DataFrame
df = pd.read_sql(query, engine)

# Define the relative path to the data folder
data_path = 'haseigel-fs/data'

# Ensure the 'data' directory exists
os.makedirs(data_path, exist_ok=True)

# Save the DataFrame to a parquet file in the specified data folder
df.to_parquet(os.path.join(data_path, 'HTML_data.parquet'))
print(f"Data saved to '{data_path}/HTML_data.parquet'")