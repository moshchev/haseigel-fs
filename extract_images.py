from bs4 import BeautifulSoup
import pandas as pd

def extract_img_attributes(html):
    # Parse the HTML content
    soup = BeautifulSoup(html, 'lxml')

    # Find all <img> tags
    img_tags = soup.find_all('img')

    # Initialize list to store img data
    img_data = []

    # Loop through each img tag and extract attributes
    for img in img_tags:
        img_attributes = img.attrs  # Get all attributes of the img tag as a dictionary
        img_data.append(img_attributes)  # Append dictionary to the list

    return img_data

if __name__ == "__main__":
    df = pd.read_parquet('data/HTML_data.parquet')
    print(extract_img_attributes(df['response_text'][0]))