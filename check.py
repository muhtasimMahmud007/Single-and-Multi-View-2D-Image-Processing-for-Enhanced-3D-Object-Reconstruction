import os

def get_filenames_without_extension(folder_path, extension):
    # Get the list of files with the specified extension
    filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.lower().endswith(extension)]
    return set(filenames)

def check_matching_filenames(png_folder, obj_folder):
    # Get the filenames without extensions from both folders
    png_filenames = get_filenames_without_extension(png_folder, '.png')
    obj_filenames = get_filenames_without_extension(obj_folder, '.obj')
    
    # Find the intersection of the two sets of filenames
    matching_filenames = png_filenames.intersection(obj_filenames)
    
    # Calculate the number of matching and mismatching filenames
    num_matching = len(matching_filenames)
    num_png_only = len(png_filenames - obj_filenames)
    num_obj_only = len(obj_filenames - png_filenames)
    num_mismatching = num_png_only + num_obj_only
    
    # Print the matching filenames and counts
    if matching_filenames:
        print("Matching filenames (without extensions):")
        for filename in matching_filenames:
            print(filename)
    else:
        print("No matching filenames found.")
    
    print(f"\nNumber of matching filenames: {num_matching}")
    print(f"Number of mismatching filenames: {num_mismatching}")
    print(f"Number of .png files without matching .obj files: {num_png_only}")
    print(f"Number of .obj files without matching .png files: {num_obj_only}")

if __name__ == "__main__":
    png_folder = "all_png"  # Replace with the path to your .png folder
    obj_folder = "recon"  # Replace with the path to your .obj folder
    
    check_matching_filenames(png_folder, obj_folder)
