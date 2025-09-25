import os
import shutil
from pathlib import Path
import yaml
from collections import Counter

original_dataset_path = Path(r"C:\Users\HP\Desktop\master_dataset")
output_dataset_path = Path(r"C:\Users\HP\Desktop\master_dataset_filtered_0.6")
original_yaml_path = original_dataset_path / "master.yaml"

high_accuracy_classes_to_keep = [
    'Healthy Wheat',
    'corn cerespora leaf spot',
    'corn common rust',
    'corn healthy',
    'potato early blight',
    'potato healthy',
    'potato late blight',
    'rice brown spot',
    'rice healthy',
    'rice hispa',
    'rice leaf blast',
    'spider mites two-spotted spider mite',
    'squash powdery mildew',
    'strawberry healthy',
    'strawberry leaf scorch',
    'tomato bacterial spot',
    'tomato early blight',
    'tomato late blight',
    'tomato leaf healthy',
    'tomato leaf mould',
    'Corn rust leaf',
    'Strawberry leaf',
    'Tomato Septoria leaf spot',
    'COW',
    'pig'
]


# --- SCRIPT LOGIC ---

def process_and_remap_dataset():
    """Main function to filter, remap, and create the new dataset."""
    
    print("🚀 Starting dataset filtering and remapping process (Threshold > 0.6)...")

    # Load original YAML to get the full class list
    try:
        with open(original_yaml_path, 'r') as f:
            original_data = yaml.safe_load(f)
        original_class_list = original_data['names']
        print(f"✅ Loaded original YAML with {len(original_class_list)} classes.")
    except Exception as e:
        print(f"❌ Error reading {original_yaml_path}: {e}")
        return

    # Create the new class list and the remapping dictionary
    final_class_list = [name for name in original_class_list if name in high_accuracy_classes_to_keep]
    final_class_map = {name: i for i, name in enumerate(final_class_list)}
    
    remapping_dict = {
        original_idx: final_class_map.get(name)
        for original_idx, name in enumerate(original_class_list)
        if name in high_accuracy_classes_to_keep
    }
    
    print(f"ℹ️  Kept {len(final_class_list)} high-accuracy classes. Discarded {len(original_class_list) - len(final_class_list)} classes.")

    # Process each split (train, valid, test)
    stats = Counter()
    for split in ['train', 'valid', 'test']:
        print(f"\n📂 Processing '{split}' split...")
        
        source_labels_path = original_dataset_path / split / "labels"
        source_images_path = original_dataset_path / split / "images"
        
        dest_labels_path = output_dataset_path / split / "labels"
        dest_images_path = output_dataset_path / split / "images"
        
        dest_labels_path.mkdir(parents=True, exist_ok=True)
        dest_images_path.mkdir(parents=True, exist_ok=True)
        
        if not source_labels_path.exists():
            print(f"   - No 'labels' directory found in '{split}', skipping.")
            continue

        for label_file in os.listdir(source_labels_path):
            if not label_file.endswith('.txt'):
                continue
            
            new_label_lines = []
            with open(source_labels_path / label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    old_class_id = int(parts[0])
                    
                    if old_class_id in remapping_dict:
                        new_class_id = remapping_dict[old_class_id]
                        new_line = f"{new_class_id} {' '.join(parts[1:])}"
                        new_label_lines.append(new_line)
            
            if new_label_lines:
                with open(dest_labels_path / label_file, 'w') as f:
                    f.write('\n'.join(new_label_lines))
                
                copied = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = source_images_path / f"{Path(label_file).stem}{ext}"
                    if image_path.exists():
                        shutil.copy(image_path, dest_images_path / image_path.name)
                        stats[split] += 1
                        copied = True
                        break
                if not copied:
                     print(f"   - ⚠️ Warning: No image found for label {label_file}")

    print("\n✅ Dataset processing complete.")
    
    # Create the new YAML file
    new_yaml_path = output_dataset_path / "master_new.yaml"
    new_yaml_data = {
        'path': str(output_dataset_path.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(final_class_list)}
    }
    
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_yaml_data, f, sort_keys=False, default_flow_style=False)
    print(f"📄 New YAML file created at: {new_yaml_path}")
    
    print("\n📊 Summary of copied images:")
    for split, count in stats.items():
        print(f"   - {split}: {count} images")

if __name__ == '__main__':
    process_and_remap_dataset()
