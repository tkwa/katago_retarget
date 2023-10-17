import os
import requests
from bs4 import BeautifulSoup

# Set base URL
BASE_URL = "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s7709731328-d3715293823/rating-games/"

# Directory where to save the SGFs
SAVE_DIR = 'sgf_downloads'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_sgf_from_page(page_number):
    response = requests.get(BASE_URL, params={'page': page_number})
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all SGF links
    sgf_links = [link['href'] for link in soup.find_all('a', text='[SGF]')]
    
    for sgf_link in sgf_links:
        sgf_link = f"https://katagotraining.org{sgf_link}"
        print(f"Trying to get {sgf_link}")
        sgf_response = requests.get(sgf_link)
        sgf_filename = os.path.join(SAVE_DIR, sgf_link.split('/')[-1])
        
        with open(sgf_filename, 'wb') as file:
            file.write(sgf_response.content)
        print(f"Downloaded {sgf_filename}")

# Let's download from page 1 to 23, change the range if required
for page in range(1, 24):
    download_sgf_from_page(page)
