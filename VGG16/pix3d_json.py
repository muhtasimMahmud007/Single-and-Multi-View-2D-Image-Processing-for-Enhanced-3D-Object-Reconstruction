import os
import json
from sklearn.model_selection import train_test_split

# Define the base directory
base_dir = 'C:\\Users\\20301470\\Downloads\\pix3d_full'

# Categories and their directories
categories = ['bed', 'bookcase', 'chair', 'desk']
img_dir = os.path.join(base_dir, 'img')
mask_dir = os.path.join(base_dir, 'mask')
model_dir = os.path.join(base_dir, 'model')

# Data structure to hold everything
data_structure = {'train': [], 'val': [], 'test': []}

# Traverse categories and create structured data
for category in categories:
    images = os.listdir(os.path.join(img_dir, category))
    images.sort()  # Ensure consistent order

    # Split data for each category
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Function to create full data paths
    def create_paths(items, split):
        result = []
        for item in items:
            img_path = os.path.join(img_dir, category, item)
            mask_path = os.path.join(mask_dir, category, item)
            model_path = os.path.join(model_dir, category, os.listdir(os.path.join(model_dir, category))[0])  # Assume one model per category
            result.append({
                'category': category,
                'split': split,
                'image': img_path,
                'mask': mask_path,
                'model': model_path
            })
        return result
    
    # Store in the main data structure
    data_structure['train'].extend(create_paths(train, 'train'))
    data_structure['val'].extend(create_paths(val, 'val'))
    data_structure['test'].extend(create_paths(test, 'test'))

# Write the data structure to a JSON file
output_json_path = os.path.join(base_dir, 'dataset_splits.json')
with open(output_json_path, 'w') as json_file:
    json.dump(data_structure, json_file, indent=4)

print("JSON file has been created at:", output_json_path)
