import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import EuroSAT

# Add project root to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.trainer import train_teacher 

# ================================
# Paths
# ================================
RAW_DIR = "data/raw"
SPLITS_DIR = "data/splits"
OUTPUT_MODEL_DIR = "outputs/models"
OUTPUT_REPORT_DIR = "outputs/reports"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

# ================================
# Load EUROSAT dataset
# ================================
def load_dataset(batch_size=64, num_workers=None):
    # Use 224x224 for ResNet training to match ImageNet pretraining resolution
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means (recommended)
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)

    # load split indices
    split_path = os.path.join(SPLITS_DIR, "train_val_test_split_seed42.json")
    with open(split_path, "r") as f:
        splits = json.load(f)

    train_set = Subset(full_dataset, splits["train_indices"])
    val_set = Subset(full_dataset, splits["val_indices"])

    # Windows compatibility: use 0 workers to avoid multiprocessing issues
    num_workers = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, len(full_dataset.classes)

# ================================
# Build Teacher Model
# ================================
def build_teacher(num_classes):
    # Use pretrained weights for transfer learning
    model = models.resnet50(pretrained=True)
    # replace head safely
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ================================
# Main Training Script
# ================================
def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load data
    train_loader, val_loader, num_classes = load_dataset(args.batch_size, num_workers=args.num_workers)

    # Build model
    model = build_teacher(num_classes).to(device)

    # Save path
    save_path = os.path.join(OUTPUT_MODEL_DIR, "teacher_resnet50.pth")

    # Train using your function
    model, history = train_teacher(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path=save_path
    )

    # Save history
    history_path = os.path.join(OUTPUT_REPORT_DIR, "teacher_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining complete!")
    print(f"Best model saved to: {save_path}")
    print(f"Training history saved to: {history_path}")


# ================================
# CLI
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Teacher (ResNet-50) on EuroSAT")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows compatibility
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
