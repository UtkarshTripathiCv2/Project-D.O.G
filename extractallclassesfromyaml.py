import yaml
from pathlib import Path

# --- CONFIGURATION ---
datasets_parent = Path(r"C:\Users\HP\Desktop\abcdesease")
# ---------------------

all_classes = set()

# Find all dataset folders
dataset_paths = [p for p in datasets_parent.iterdir() if p.is_dir()]

for path in dataset_paths:
    yaml_file = path / 'data.yaml'
    if yaml_file.exists():
        with open(yaml_file, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if 'names' in data and isinstance(data['names'], list):
                    for name in data['names']:
                        all_classes.add(str(name).strip())
            except yaml.YAMLError as e:
                print(f"Error reading {yaml_file}: {e}")

# Print the final, sorted list
sorted_classes = sorted(list(all_classes))
print("master_class_list = [")
for name in sorted_classes:
    print(f"    '{name}',")
print("]")
