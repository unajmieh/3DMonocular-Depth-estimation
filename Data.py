import zipfile
import os

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# Example usage
zip_path = 'path_to_your_zip_file.zip'
extract_to = 'path_to_extract_directory'
extract_zip("./Dataset/CamSeq01.zip", "./Dataset")