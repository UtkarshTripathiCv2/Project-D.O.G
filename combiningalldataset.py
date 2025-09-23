import os
import shutil

# --- CONFIGURATION ---

# 1. Define the paths to your original datasets
dataset_paths = {
    "fire_smoke": "/path/to/your/FireSmoke_Dataset",
    "chilli": "/path/to/your/Chilli_Dataset",
    "tomato": "/path/to/your/Tomato_Dataset"
}

# 2. Define the path for the new combined dataset
output_path = "/path/to/your/Combined_Dataset"

# 3. Define the master class mapping.
# This is the MOST IMPORTANT step. It tells the script how to convert old class IDs to new ones.
# Format: "dataset_name": {old_id_0: new_id_A, old_id_1: new_id_B, ...}
CLASS_MAPPING = {
    "fire_smoke": {
        0: 0,  # fire -> fire
        1: 1   # smoke -> smoke
    },
    "chilli": {
        0: 2,  # chilli_leaf_spot -> chilli_leaf_spot
        1: 3   # chilli_powdery_mildew -> chilli_powdery_mildew
        # Add more chilli classes here if you have them
    },
    "tomato": {
        0: 4,  # tomato_late_blight -> tomato_late_blight
        1: 5   # tomato_leaf_mold -> tomato_leaf_mold
        # Add more tomato classes here if you have them
    }
}

# --- SCRIPT LOGIC ---

def process_and_copy_files(dataset_name, source_base_path, dest_base_path, mapping):
    """
    Processes a single dataset: remaps class IDs in label files and copies
    both images and labels to the destination directory.
    """
    print(f"--- Processing dataset: {dataset_name} ---")
    
    for split in ["train", "val", "test"]:
        source_label_dir = os.path.join(source_base_path, split, "labels")
        source_image_dir = os.path.join(source_base_path, split, "images")
        
        dest_label_dir = os.path.join(dest_base_path, split, "labels")
        dest_image_dir = os.path.join(dest_base_path, split, "images")
        
        if not os.path.exists(source_label_dir):
            print(f"Warning: Directory not found, skipping: {source_label_dir}")
            continue

        print(f"Processing split: {split}")
        
        for label_filename in os.listdir(source_label_dir):
            if not label_filename.endswith(".txt"):
                continue

            # --- Remap Class IDs in Label File ---
            new_label_content = []
            with open(os.path.join(source_label_dir, label_filename), 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    old_class_id = int(parts[0])
                    
                    if old_class_id in mapping:
                        new_class_id = mapping[old_class_id]
                        new_line = f"{new_class_id} {' '.join(parts[1:])}"
                        new_label_content.append(new_line)
                    else:
                        print(f"Warning: Class ID {old_class_id} not in mapping for {dataset_name}. Skipping line.")

            # Write the new label file to the destination
            with open(os.path.join(dest_label_dir, label_filename), 'w') as f_out:
                f_out.write("\n".join(new_label_content))
            
            # --- Copy the Corresponding Image ---
            image_filename_base = os.path.splitext(label_filename)[0]
            # Find the correct image extension (.jpg, .png, etc.)
            for ext in ['.jpg', '.jpeg', '.png']:
                source_image_path = os.path.join(source_image_dir, image_filename_base + ext)
                if os.path.exists(source_image_path):
                    shutil.copy(source_image_path, dest_image_dir)
                    break

def main():
    # Create the main output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)

    # Process each dataset defined in the config
    for name, path in dataset_paths.items():
        if name not in CLASS_MAPPING:
            print(f"Error: No class mapping found for dataset '{name}'. Skipping.")
            continue
        process_and_copy_files(name, path, output_path, CLASS_MAPPING[name])

    print("\n--- All datasets processed successfully! ---")
    print(f"Combined dataset is ready at: {output_path}")


if __name__ == "__main__":
    main()
```

### How to Use the Script

1.  **Save the Code:** Save the script above as a Python file, for example, `combine_datasets.py`.
2.  **Configure the Paths:** Open the file and edit the `dataset_paths` and `output_path` variables to match the locations on your computer.
3.  **Configure the Class Mapping:** This is the most important part. Carefully edit the `CLASS_MAPPING` dictionary to match the new master list you created in Step 1. Make sure every class from every dataset has a new, unique ID.
4.  **Run the Script:** Open a terminal or command prompt, navigate to where you saved the file, and run it:
    ```bash
    python combine_datasets.py
    
