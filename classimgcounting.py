import yaml
from pathlib import Path
from collections import Counter

# --- CONFIGURATION ---

# 1. Set the path to your main dataset folder (e.g., master_dataset or master_dataset_filtered)
DATASET_PATH = Path(r"C:\Users\HP\Desktop\master_dataset")

# 2. Set the name of your YAML file
YAML_FILENAME = "master.yaml"


# --- SCRIPT LOGIC ---

def count_class_instances(dataset_path, yaml_filename):
    """Counts instances of each class in a YOLO dataset."""
    
    yaml_path = dataset_path / yaml_filename
    
    # Load the YAML file to get class names
    try:
        # CORRECTED: Changed 'in f' to 'as f' on the line below
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Handle both list and dictionary formats for names
        if isinstance(data['names'], list):
            class_names = {i: name for i, name in enumerate(data['names'])}
        else:
            class_names = data['names']
        print(f"✅ Found {len(class_names)} classes in {yaml_filename}")
    except Exception as e:
        print(f"❌ Error: Could not read or parse {yaml_path}. Details: {e}")
        return

    class_counts = Counter()
    image_counts = Counter()
    
    # Iterate through train, valid, and test splits
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / 'labels'
        
        if not labels_dir.is_dir():
            print(f"⚠️  Warning: No 'labels' directory found for '{split}' split. Skipping.")
            continue
        
        # Count images by counting label files
        label_files = list(labels_dir.glob('*.txt'))
        image_counts[split] = len(label_files)
        
        # Count class instances within the label files
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    try:
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
                    except (ValueError, IndexError):
                        # Handle empty or malformed lines
                        pass

    # --- Print the Report ---
    print("\n" + "="*40)
    print("📊 DATASET ANALYSIS REPORT")
    print("="*40)

    print("\nTotal Images per Split:")
    for split, count in image_counts.items():
        if count > 0:
            print(f"  - {split.capitalize()}:\t{count} images")

    print("\nTotal Instances per Class (across all splits):")
    for class_id, class_name in sorted(class_names.items()):
        count = class_counts.get(class_id, 0)
        print(f"  - ID {class_id:2d} | {class_name:<40} | {count} instances")
        
    print("="*40)


if __name__ == '__main__':
    count_class_instances(DATASET_PATH, YAML_FILENAME)
