import os
import sys

# Ensure the project root (the folder containing `src`) is on sys.path so
# `from src...` imports work when running this script directly.
# This inserts the lightvision project root (two levels up from this file)
# at the front of sys.path.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
import json

from src.data.preprocessing import get_train_transform
from src.training.trainer import train_teacher
from torchvision.datasets import EuroSAT

if __name__ == '__main__':
    # Directories
    RAW_DIR = 'data/raw'
    SPLITS_DIR = 'data/splits'
    OUTPUT_DIR = 'outputs/models'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load split indices
    with open(os.path.join(SPLITS_DIR, 'train_val_test_split_seed42.json'), 'r') as f:
        splits = json.load(f)

    # Dataset and transforms
    transform = get_train_transform(img_size=64)
    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)

    train_dataset = Subset(full_dataset, splits['train_indices'])
    val_dataset = Subset(full_dataset, splits['val_indices'])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Teacher model
    teacher_model = models.resnet50(pretrained=True)
    num_features = teacher_model.fc.in_features
    teacher_model.fc = nn.Linear(num_features, len(full_dataset.classes))
    teacher_model = teacher_model.to(device)

    # Training parameters
    epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-4

    # Train teacher
    trained_teacher, history = train_teacher(teacher_model, train_loader, val_loader, device,
                                             epochs=epochs, lr=learning_rate, weight_decay=weight_decay,
                                             save_path=os.path.join(OUTPUT_DIR, 'teacher_resnet50.pth'))

    # Optionally, save training history
    import pickle
    with open(os.path.join(OUTPUT_DIR, 'teacher_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print("Teacher model trained and saved.")
