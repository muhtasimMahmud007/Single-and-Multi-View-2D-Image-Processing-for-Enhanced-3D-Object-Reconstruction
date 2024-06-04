import os

def remove_suffix_from_obj_files(folder_path, suffix='_256'):
    # Iterate through the files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is an .obj file and contains the suffix
        if filename.lower().endswith('.obj') and suffix in filename:
            # Remove the suffix
            new_filename = filename.replace(suffix, '')
            # Get the full paths
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} to {new_filename}')

if __name__ == "__main__":
    folder_path = "recon"  # Replace with the path to your folder containing .obj files
    remove_suffix_from_obj_files(folder_path)
