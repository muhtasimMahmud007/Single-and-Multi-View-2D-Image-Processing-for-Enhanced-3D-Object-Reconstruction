import os
import json
from sklearn.model_selection import train_test_split

def split_folders(folder_path, taxonomy_id, taxonomy_name):
    # List all directories in the specified folder
    folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    if not folders:
        print(f"No folders found in the specified folder: {folder_path}")
        return
    
    print(f"Folders found: {folders}")

    # Split into train, val, and test
    train_folders, temp_folders = train_test_split(folders, test_size=0.3, random_state=42)  # 70% train, 30% temp
    val_folders, test_folders = train_test_split(temp_folders, test_size=0.5, random_state=42)  # 15% val, 15% test
    
    data = {
        "taxonomy_id": taxonomy_id,
        "taxonomy_name": taxonomy_name,
        "baseline": {
            "1-view": 0.6,
            "2-view": 0.65,
            "3-view": 0.68,
            "4-view": 0.7,
            "5-view": 0.72
        },
        "train": train_folders,
        "val": val_folders,
        "test": test_folders
    }
    
    output_file = os.path.join(folder_path, "data_split.json")
    
    # Save the data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Ensure to use the correct folder path
    folder_path = r"D:\plusplus\dataset\2002"  # Replace with the correct path to your folder containing subfolders
    taxonomy_id = "recon"
    taxonomy_name = "human"
    
    split_folders(folder_path, taxonomy_id, taxonomy_name)
