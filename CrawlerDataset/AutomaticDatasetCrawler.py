import os  
import requests  
from urllib.parse import urlparse, urljoin  
from bs4 import BeautifulSoup  

def download_image(url, folder):  
    """Download images from the URLs."""  
    try:  
        if not urlparse(url).scheme in ['http', 'https']:  
            print(f"Invalid URL: {url}")  
            return  

        response = requests.get(url, stream=True)  
        response.raise_for_status()  # Check for request errors  

        file_name = os.path.basename(urlparse(url).path)  
        file_path = os.path.join(folder, file_name)  

        # Handle name conflicts  
        base, extension = os.path.splitext(file_name)  
        counter = 1  
        while os.path.exists(file_path):  
            file_path = os.path.join(folder, f"{base}_{counter}{extension}")  
            counter += 1  
        
        with open(file_path, 'wb') as img_file:  
            for chunk in response.iter_content(1024):  
                img_file.write(chunk)  

        print(f"Downloaded: {file_path}")  
    except Exception as e:  
        print(f"Error downloading {url}: {e}")  

def find_image_urls(page_url):  
    """Find all image URLs on a given webpage."""  
    image_urls = []  
    try:  
        response = requests.get(page_url)  
        response.raise_for_status()  # Check for request errors  
        soup = BeautifulSoup(response.text, 'html.parser')  

        # Find all <img> tags and extract the src attribute  
        for img in soup.find_all('img'):  
            img_url = img.get('src')  
            if img_url and is_image_url(img_url):  
                img_url = urljoin(page_url, img_url)  # Resolve relative URLs  
                image_urls.append(img_url)  
    except Exception as e:  
        print(f"Error fetching images from {page_url}: {e}")  

    return image_urls  

def is_image_url(url):  
    """Check if the URL is an image based on its extension."""  
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']  
    return any(url.lower().endswith(ext) for ext in allowed_extensions)  

def crawl_image_urls(page_url, download_folder):  
    """Crawl the given webpage and download images."""  
    if not os.path.exists(download_folder):  
        os.makedirs(download_folder)  

    image_urls = find_image_urls(page_url)  
    for url in image_urls:  
        download_image(url, download_folder)  

# Example usage  
webpage_url = "https://www.wikipedia.org"  
download_folder = "downloaded_images"  

# Start crawling and downloading images  
crawl_image_urls(webpage_url, download_folder)
