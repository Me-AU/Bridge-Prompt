import shutil

def zip_folder(folder_path, output_zip_path):
    """
    Zips the contents of a folder.
    
    Args:
        folder_path (str): Path to the folder to be zipped.
        output_zip_path (str): Path where the output zip file should be saved (without .zip extension).
    """
    shutil.make_archive(output_zip_path, 'zip', folder_path)
    print(f"Folder '{folder_path}' successfully zipped to '{output_zip_path}.zip'")

# Example usage:
folder_path = '/home/user/visisonrd-action-segmentation/Bridge-Prompt/data/gtea'
output_zip_path = '/home/user/visisonrd-action-segmentation/Bridge-Prompt/data/gtea'
zip_folder(folder_path, output_zip_path)
