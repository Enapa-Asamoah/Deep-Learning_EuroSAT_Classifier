import os
import sys

# CRITICAL: Disable user site-packages to avoid broken PyTorch installation
# in C:\Users\...\AppData\Roaming\Python\... that conflicts with conda env
import site
site.ENABLE_USER_SITE = False
sys.path = [p for p in sys.path if 'AppData\\Roaming\\Python' not in p]

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.datasets import EuroSAT
from torchvision import transforms

# Add parent directory to path to resolve src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import your evaluator
from src.evaluation.evaluator import evaluate_model


# ================================
# Paths
# ================================
RAW_DIR = "data/raw"
SPLITS_DIR = "data/splits"
OUTPUT_MODEL_DIR = "outputs/models"
PLOTS_DIR = "outputs/plots"
REPORTS_DIR = "outputs/reports"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ================================
# Load EuroSAT dataset + splits
# ================================
def load_dataset(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.344, 0.380, 0.408],
                             std=[0.180, 0.166, 0.160])
    ])

    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)

    # load split indices
    with open(os.path.join(SPLITS_DIR, "train_val_test_split_seed42.json"), "r") as f:
        splits = json.load(f)

    test_dataset = Subset(full_dataset, splits["test_indices"])

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return test_loader, full_dataset.classes


# ================================
# Build Model (teacher/student/compressed)
# ================================
def build_model(model_type, num_classes, device):
    if model_type == "resnet50_teacher":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_type == "mobilenet_v3":
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_type == "compressed":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    else:
        raise ValueError("Unknown model type selected.")

    return model.to(device)


# ================================
# Main Evaluation Script
# ================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    test_loader, class_names = load_dataset(batch_size=args.batch_size)

    # build model
    model = build_model(args.model_type, len(class_names), device)

    # load checkpoint
    checkpoint_path = os.path.join(OUTPUT_MODEL_DIR, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Perform evaluation
    metrics, y_true, y_pred, plot_paths = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        classes=class_names,
        history=None,
        output_dir=PLOTS_DIR,
        model_name=args.model_type
    )

    # Save metrics JSON
    results_path = os.path.join(REPORTS_DIR, f"{args.model_type}_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation complete. Results saved to {results_path}")


# ================================
# CLI Interface
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained or compressed models")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resnet50_teacher", "mobilenet_v2", "mobilenet_v3", "compressed"]
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Filename of the .pth checkpoint under outputs/models/"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    args = parser.parse_args()
    main(args)
