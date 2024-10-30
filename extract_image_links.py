import re
import pandas as pd
import os

df = pd.read_parquet('haseigel-fs/data/HTML_data.parquet')

def extract_image_links(html):
    # Use regular expressions to find all image links in the HTML
    image_links = re.findall(r'<img[^>]+src=[\'"]([^\'"]+)[\'"]', html)
    background_images = re.findall(r'style=[\'"][^\'"]*background-image:\s*url\([\'"]?([^\)]+?)[\'"]?\)', html)
    css_images = re.findall(r'url\([\'"]?([^\)]+?)[\'"]?\)', html)
    
    # Return a dictionary with separate categories
    return {
        'image_links': list(dict.fromkeys(image_links)),
        'background_images': list(dict.fromkeys(background_images)),
        'css_images': list(dict.fromkeys(css_images))
    }

# Extract all image links from the response_text of all rows
all_image_links = {
    'image_links': [],
    'background_images': [],
    'css_images': []
}

for html in df['response_text']:
    links = extract_image_links(html)
    all_image_links['image_links'].extend(links['image_links'])
    all_image_links['background_images'].extend(links['background_images'])
    all_image_links['css_images'].extend(links['css_images'])

# Remove duplicates from each category
for category in all_image_links:
    all_image_links[category] = list(dict.fromkeys(all_image_links[category]))

# # Print the first 5 links from each category
# print("Image links:", all_image_links['image_links'][:5])
# print("Background images:", all_image_links['background_images'][:5])
# print("CSS images:", all_image_links['css_images'][:5])

# Convert the dictionary to a DataFrame
df_image_links = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_image_links.items()]))

# Save the DataFrame to a Parquet file
df_image_links.to_parquet('haseigel-fs/data/extracted_image_links.parquet')