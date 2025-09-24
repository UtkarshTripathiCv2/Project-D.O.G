import os
import shutil
from pathlib import Path
from collections import defaultdict

datasets_parent = Path(r"C:\Users\HP\Desktop\datasets_all")

master_class_list = [
    'Black Rust', 'Brown Rust', 'Healthy Wheat', 'Yellow Rust', 'apple black rot',
    'apple healthy', 'apple scab', 'bell pepper bacterial spot', 'bell pepper healthy',
    'cassava Brown Streak Disease', 'cassava bacterial blight', 'cassava green mottle',
    'cedar apple rust', 'cherry healthy', 'cherry powdery mildew', 'corn cerespora leaf spot',
    'corn common rust', 'corn healthy', 'grape black rot', 'grape esca', 'grape healthy',
    'grape leaf blight', 'healthy cassava', 'mosaic cassava', 'northern leaf blight',
    'orange citrus greening', 'peach bacterial spot', 'peach healthy', 'potato early blight',
    'potato healthy', 'potato late blight', 'rice brown spot', 'rice healthy', 'rice hispa',
    'rice leaf blast', 'spider mites two-spotted spider mite', 'squash powdery mildew',
    'strawberry healthy', 'strawberry leaf scorch', 'tomato bacterial spot',
    'tomato early blight', 'tomato late blight', 'tomato leaf healthy', 'tomato leaf mould',
    'tomato septoria leaf spot', 'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf',
    'Bell_pepper leaf spot', 'Bell_pepper leaf', 'Blueberry leaf', 'Cherry leaf',
    'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Peach leaf',
    'Potato leaf early blight', 'Potato leaf late blight', 'Potato leaf', 'Raspberry leaf',
    'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf bacterial spot',
    'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus',
    'Tomato leaf', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf',
    'grape leaf black rot', 'grape leaf', 'chilli0', 'chilli1', 'chilli2', 'chilli3',
    'chilli4', 'chilli5', 'chilli6', 'chilli7', 'COW', 'horse', 'pig', 'sheep', 'undefined'
]


output_path = Path(r"C:\Users\HP\Desktop\master_dataset")



master_class_map = {name.lower().strip(): i for i, name in enumerate(master_class_list)}


stats = defaultdict(lambda: defaultdict(int))

def get_class_list_from_yaml(yaml_path):
    """Reads a YOLO data.yaml file and returns the list of class names."""
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                return [str(name).strip() for name in data['names']]
    except ImportError:
        print("⚠ PyYAML not installed. Run: pip install PyYAML")
    except Exception as e:
        print(f"⚠ Error reading {yaml_path}: {e}")
    return []

def remap_and_copy_files(original_path, split, old_classes):
    """Reads label files, remaps class indices, and copies images/labels into master dataset."""
    image_dir = original_path / split / 'images'
    label_dir = original_path / split / 'labels'

    if not label_dir.is_dir() or not image_dir.is_dir():
        print(f"   ⚠ Skipping '{split}' in {original_path.name} (missing dirs).")
        return

    dest_img_dir = output_path / split / 'images'
    dest_lbl_dir = output_path / split / 'labels'
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    dataset_prefix = original_path.name  # Use dataset folder name for uniqueness

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        new_labels = []
        with open(label_dir / label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls_id = int(parts[0])

                if old_cls_id < len(old_classes):
                    class_name = old_classes[old_cls_id].strip().lower()
                    if class_name in master_class_map:
                        new_cls_id = master_class_map[class_name]
                        new_line = f"{new_cls_id} {' '.join(parts[1:])}"
                        new_labels.append(new_line)
                    else:
                        print(f"     ⚠ Unknown class '{old_classes[old_cls_id]}' in {label_file}")
                else:
                    print(f"     ⚠ Invalid class index {old_cls_id} in {label_file}")

        if new_labels:
            # Unique filenames
            new_label_file = f"{dataset_prefix}_{label_file}"
            new_image_name = f"{dataset_prefix}_{Path(label_file).stem}"

            with open(dest_lbl_dir / new_label_file, 'w') as f:
                f.write('\n'.join(new_labels))

            copied = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                image_path = image_dir / f"{Path(label_file).stem}{ext}"
                if image_path.exists():
                    shutil.copy(image_path, dest_img_dir / f"{new_image_name}{ext}")
                    copied = True
                    stats[dataset_prefix][split] += 1
                    break
            if not copied:
                print(f"     ⚠ No image found for {label_file}")

def create_master_yaml():
    """Creates final master.yaml for YOLO training."""
    yaml_path = output_path / "master.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {output_path}/train/images\n")
        f.write(f"val: {output_path}/valid/images\n")
        f.write(f"test: {output_path}/test/images\n\n")
        f.write("names:\n")
        for i, name in enumerate(master_class_list):
            f.write(f"  {i}: {name}\n")
    print(f"\n📄 Master YAML created at: {yaml_path}")

if __name__ == '__main__':
    print("🚀 Starting dataset merge + remap...")

    # Auto-detect dataset folders inside parent directory
    dataset_paths = [p for p in datasets_parent.iterdir() if p.is_dir()]

    for dataset_path in dataset_paths:
        yaml_file = dataset_path / 'data.yaml'

        if not yaml_file.exists():
            print(f"❌ No data.yaml in {dataset_path}, skipping.")
            continue

        print(f"\n📂 Processing dataset: {dataset_path.name}")
        old_class_list = get_class_list_from_yaml(yaml_file)
        if not old_class_list:
            print(f"❌ Could not read classes from {yaml_file}, skipping.")
            continue

        print(f"   Found {len(old_class_list)} classes (sample: {old_class_list[:5]})")

        for split in ['train', 'valid', 'test']:
            print(f"   → Remapping '{split}'...")
            remap_and_copy_files(dataset_path, split, old_class_list)

    create_master_yaml()

    print("\n✅ All done! Master dataset ready in:", output_path)

    print("\n📊 Summary Report:")
    for ds, splits in stats.items():
        split_counts = {s: c for s, c in splits.items()}
        print(f"  {ds}: {split_counts}")
