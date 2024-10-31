from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve

def extract_img_attributes(html):
    # Parse the HTML content
    soup = BeautifulSoup(html, 'lxml')

    # Find all <img> tags
    img_tags = soup.find_all('img')

    # Initialize list to store each img tag's attributes as dictionaries
    img_data = []

    # Loop through each img tag and extract attributes
    for img in img_tags:
        img_attributes = img.attrs  # Get all attributes of the img tag as a dictionary
        img_data.append(img_attributes)  # Append dictionary to the list

    return img_data

def save_combined_html(df, output_file="haseigel-fs/data/combined.html"):
    # Combine all response text into one HTML file
    with open(output_file, "w", encoding="utf-8") as file:
        for html_content in df["response_text"]:
            file.write(html_content)
            file.write("\n")  # Separate each HTML content by a newline for readability

def download_images_with_local_path(dict_list, download_folder="haseigel-fs/data/images"):
    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)
    
    # Loop through each dictionary in the list
    for index, img_data in enumerate(dict_list):
        img_url = img_data.get("src")
        
        # Check if img_url exists and is a valid URL (not a data URI)
        if img_url and urlparse(img_url).scheme in ["http", "https"]:
            # Generate a unique filename for each image
            img_name = f"{index}_{os.path.basename(urlparse(img_url).path)}"
            img_path = os.path.join(download_folder, img_name)
            
            # Download the image
            try:
                urlretrieve(img_url, img_path)
                print(f"Downloaded: {img_name}")
                
                # Add the relative path to the dictionary
                img_data["local_path"] = os.path.relpath(img_path, start=download_folder)
                
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")
        else:
            print(f"Skipping item {index} - No valid image URL found.")

if __name__ == "__main__":
    # Load the parquet file
    df = pd.read_parquet('haseigel-fs/data/HTML_data.parquet')

    # Save all HTML content into one combined HTML file
    save_combined_html(df)

    # Read combined HTML for parsing
    with open("haseigel-fs/data/combined.html", "r", encoding="utf-8") as file:
        combined_html = file.read()

    # Extract image attributes into a dictionary list
    img_data = extract_img_attributes(combined_html)

    # Download images and add local paths to the dictionary list
    download_images_with_local_path(img_data)

    # Optionally, convert img_data to a DataFrame and save
    img_df = pd.DataFrame(img_data)
    img_df.to_csv("data/image_attributes_with_local_path.csv", index=False)  # Save to CSV for further analysis