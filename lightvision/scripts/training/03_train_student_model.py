import os
import sys

# Ensure the project root (the folder containing `src`) is on sys.path so
# `from src...` imports work when running this script directly.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
import json

from src.training.student_trainer import train_student

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

    # Dataset
    from torchvision.datasets import EuroSAT
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.344,0.380,0.408], std=[0.180,0.166,0.160])
    ])
    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)
    train_dataset = Subset(full_dataset, splits['train_indices'])
    val_dataset = Subset(full_dataset, splits['val_indices'])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Student architectures to train
    student_architectures = {
        'mobilenet_v2': models.mobilenet_v2(pretrained=True),
        'mobilenet_v3': models.mobilenet_v3_small(pretrained=True),
        'shufflenet_v2': models.shufflenet_v2_x1_0(pretrained=True)
    }

    for name, model in student_architectures.items():
        # Modify classifier for EuroSAT
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, len(full_dataset.classes))
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, len(full_dataset.classes))
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, len(full_dataset.classes))

        model = model.to(device)

        # Training parameters
        epochs = 10
        learning_rate = 1e-4
        weight_decay = 1e-4

        # Train student
        trained_student, history = train_student(model, train_loader, val_loader, device,
                                                epochs=epochs, lr=learning_rate, weight_decay=weight_decay,
                                                save_path=os.path.join(OUTPUT_DIR, f'{name}.pth'))

        # Save training history
        import pickle
        with open(os.path.join(OUTPUT_DIR, f'{name}_history.pkl'), 'wb') as f:
            pickle.dump(history, f)

        print(f"Student model {name} trained and saved.")