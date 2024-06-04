# import os
# import shutil

# def organize_obj_files(folder_path):
#     # Iterate through the files in the specified folder
#     for filename in os.listdir(folder_path):
#         # Check if the file is an .obj file
#         if filename.lower().endswith('.obj'):
#             # Remove the .obj extension to get the folder name
#             folder_name = os.path.splitext(filename)[0]
#             # Create the new folder path
#             new_folder_path = os.path.join(folder_path, folder_name)
#             # Create the folder if it doesn't exist
#             if not os.path.exists(new_folder_path):
#                 os.makedirs(new_folder_path)
#             # Get the full paths for the old file and the new file
#             old_file_path = os.path.join(folder_path, filename)
#             new_file_path = os.path.join(new_folder_path, 'model.obj')
#             # Move and rename the file
#             shutil.move(old_file_path, new_file_path)
#             print(f'Moved and renamed: {filename} to {new_file_path}')

# if __name__ == "__main__":
#     folder_path = "recon"  # Replace with the path to your folder containing .obj files
#     organize_obj_files(folder_path)



import os
import shutil

def organize_png_files(folder_path):
    # Iterate through the files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .png file
        if filename.lower().endswith('.png'):
            # Remove the .png extension to get the folder name
            folder_name = os.path.splitext(filename)[0]
            # Create the new folder path
            new_folder_path = os.path.join(folder_path, folder_name)
            # Create the folder if it doesn't exist
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            # Get the full paths for the old file and the new file
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(new_folder_path, '00.png')
            # Move and rename the file
            shutil.move(old_file_path, new_file_path)
            print(f'Moved and renamed: {filename} to {new_file_path}')

if __name__ == "__main__":
    folder_path = "all_png"  # Replace with the path to your folder containing .png files
    organize_png_files(folder_path)

