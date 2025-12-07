import os
# Workaround for OpenMP runtime conflicts on Windows when multiple OpenMP
# runtimes are present (libomp.dll vs libiomp5md.dll). This allows the
# process to continue. This is an unsafe, temporary workaround — see
# recommendations below for long-term fixes.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# Reduce thread contention while running preprocessing
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
import torch
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import random_split

# Directories
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
SPLITS_DIR = 'data/splits'

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.344, 0.380, 0.408], std=[0.180, 0.166, 0.160])  # EuroSAT RGB stats
])

# Download EuroSAT RGB dataset
dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)

# Train/val/test split (70/15/15)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

torch.manual_seed(42)  # for reproducibility
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Save split indices
splits = {
    'train_indices': train_dataset.indices,
    'val_indices': val_dataset.indices,
    'test_indices': test_dataset.indices
}

with open(os.path.join(SPLITS_DIR, 'train_val_test_split_seed42.json'), 'w') as f:
    json.dump(splits, f)

print(f"Dataset prepared and splits saved to {SPLITS_DIR}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
