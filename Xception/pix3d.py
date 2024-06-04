import os
import json
from sklearn.model_selection import train_test_split

# Base directory where categories are stored
base_dir = 'C:\\Users\\20301470\\Downloads\\pix3d_full'

# List of categories you want to process
categories = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']  # Add or remove categories as needed

dataset_structure = []

# Function to check and collect image files
def collect_image_files(directory):
    valid_extensions = ['.jpg', '.png']  # List of valid image file extensions
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_extensions]
    return files

# Process each category
for category in categories:
    # Directory for this category's images
    img_dir = os.path.join(base_dir, 'img', category)
    
    # Check if directory exists and read file names
    if os.path.exists(img_dir):
        file_ids = [os.path.splitext(f)[0] for f in collect_image_files(img_dir)]
        if len(file_ids) == 0:
            print(f"No image files found in {img_dir}. Skipping this category.")
            continue  # Skip this category if no images are found

        # Split data into training, validation, and testing sets
        train_ids, test_val_ids = train_test_split(file_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)
        
        # Structure for this category
        category_data = {
            "taxonomy_id": category,
            "taxonomy_name": category,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }
        
        # Append this category's data to the overall dataset structure
        dataset_structure.append(category_data)
    else:
        print(f"Directory {img_dir} does not exist. Please check the path.")

# Path for the JSON output file
output_json_path =  'C:\\Users\\20301470\\Downloads\\pix3d_full\\your_dataset_structure.json'

# Write the structured data to a JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(dataset_structure, json_file, indent=4)

print("JSON file has been created at:", output_json_path)
