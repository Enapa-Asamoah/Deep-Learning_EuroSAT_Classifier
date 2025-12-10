import os
import sys
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import EuroSAT

# Ensure project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.student_trainer import train_student  # your custom trainer

if __name__ == '__main__':
    # ================================
    # Directories and Device
    # ================================
    RAW_DIR = 'data/raw'
    SPLITS_DIR = 'data/splits'
    OUTPUT_DIR = 'outputs/models'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # DirectML has poor performance, use CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # ================================
    # Load dataset splits
    # ================================
    with open(os.path.join(SPLITS_DIR, 'train_val_test_split_seed42.json'), 'r') as f:
        splits = json.load(f)

    # Use 224x224 for student too (match teacher / pretrained)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)
    train_dataset = Subset(full_dataset, splits['train_indices'])
    val_dataset = Subset(full_dataset, splits['val_indices'])

    # Disable multiprocessing for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)


    #Build student model (ResNet18)
    student_name = 'student_resnet18'
    student_model = models.resnet18(pretrained=True)

    
    if hasattr(student_model, 'fc'):
        in_features = student_model.fc.in_features
        student_model.fc = nn.Linear(in_features, len(full_dataset.classes))
    else:
        student_model.classifier[-1] = nn.Linear(student_model.classifier[-1].in_features, len(full_dataset.classes))

    student_model = student_model.to(device)

    # ================================
    # Training parameters
    # ================================
    epochs = 10
    learning_rate = 1e-4
    weight_decay = 1e-4

    # ================================
    # Train student
    # ================================
    trained_model, history = train_student(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        save_path=os.path.join(OUTPUT_DIR, f'{student_name}.pth')
    )

    # Save training history
    history_path = os.path.join(OUTPUT_DIR, f'{student_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"Student model {student_name} trained. Model and history saved to {OUTPUT_DIR}.")
