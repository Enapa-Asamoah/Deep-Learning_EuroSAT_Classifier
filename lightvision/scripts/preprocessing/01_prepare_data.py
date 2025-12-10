import os
import json
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import EuroSAT

# ================================
# Directories
# ================================
RAW_DIR = 'data/raw'
SPLITS_DIR = 'data/splits'

os.makedirs(SPLITS_DIR, exist_ok=True)

# ================================
# Load dataset WITHOUT transforms
# (to avoid data leakage)
# ================================
print("Downloading EuroSAT dataset (no transforms applied)...")
dataset = EuroSAT(root=RAW_DIR, download=True)

num_samples = len(dataset)
print(f"Total samples: {num_samples}")

# ================================
# Extract labels for stratified split
# ================================
labels = [dataset[i][1] for i in range(num_samples)]

# ================================
# Stratified 70/15/15 split
# ================================
train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    list(range(num_samples)),
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,   # split 15/15
    stratify=temp_labels,
    random_state=42
)

print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

# ================================
# Save splits
# ================================
splits = {
    'train_indices': train_indices,
    'val_indices': val_indices,
    'test_indices': test_indices
}

output_path = os.path.join(SPLITS_DIR, 'train_val_test_split_seed42.json')

with open(output_path, 'w') as f:
    json.dump(splits, f, indent=4)

print(f"Saved split file to {output_path}")
