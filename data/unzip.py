import zipfile

# Replace 'path/to/your/file.zip' with the path to your zip file
zip_file_path = '/home/user/visisonrd-action-segmentation/Bridge-Prompt/data/gtea.zip'
extract_to_path = '/home/user/visisonrd-action-segmentation/Bridge-Prompt/data/gtea'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Extracted all files to {extract_to_path}")