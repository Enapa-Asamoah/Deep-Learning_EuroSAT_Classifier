import os
import sys
import json
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import EuroSAT

# ================================
# Path setup
# ================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

RAW_DIR = 'data/raw'
SPLITS_DIR = 'data/splits'
OUTPUT_DIR = 'outputs/models'
PLOTS_DIR = 'outputs/plots'
REPORTS_DIR = 'outputs/reports'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================
# Preprocessing config
# ================================
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ================================
# Dataset loader
# ================================
def load_dataset(batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)

    split_path = os.path.join(SPLITS_DIR, 'train_val_test_split_seed42.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError("Run 01_prepare_data.py first.")

    with open(split_path, 'r') as f:
        splits = json.load(f)

    val_dataset = Subset(full_dataset, splits['val_indices'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    return val_loader, len(full_dataset.classes)

# ================================
# Model factory
# ================================
def build_model(name, num_classes):
    name = name.lower()

    # Teacher is ResNet50
    if "teacher" in name:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Pruned model is ResNet50 (from teacher)
    if "pruned" in name:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Student and its compressions (distilled, qat) are ResNet18
    if "resnet18" in name or "student" in name or "distilled" in name or "qat" in name:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if "resnet50" in name:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if "mobile" in name:
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    # default student
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ================================
# Robust checkpoint loader
# ================================
def load_checkpoint(model_name, ckpt_path, num_classes):
    model = build_model(model_name, num_classes).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "weights"]:
            if key in ckpt:
                ckpt = ckpt[key]
                break

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print("  Missing keys:", missing[:5])
    if unexpected:
        print("  Unexpected keys:", unexpected[:5])

    return model

# ================================
# Evaluation function
# ================================
def evaluate(model, dataloader):
    model.eval()  # Ensure eval mode
    
    # Disable gradient computation and BatchNorm training
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        total_correct = 0
        total = 0

        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)

            total_loss += loss.item() * img.size(0)
            _, preds = out.max(1)
            total_correct += preds.eq(label).sum().item()
            total += label.size(0)

    acc = total_correct / total
    loss = total_loss / total
    return acc, loss

# ================================
# Latency test
# ================================
def measure_latency(model, runs=50):
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.time() - start) / runs * 1000

# ================================
# Report plotting
# ================================
def plot_metric(names, values, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

# ================================
# Main
# ================================
def main(args):
    val_loader, num_classes = load_dataset(args.batch_size, args.num_workers)

    checkpoints = {
        "teacher_resnet50": args.teacher,
        "student_resnet18": args.student,
        "pruned_model": args.pruned,
        "distilled_model": args.distilled,
        "qat_model": args.qat
    }

    results = {}

    for name, ckpt in checkpoints.items():
        if ckpt is None or not os.path.exists(ckpt):
            print(f"Skipping {name}, file missing.")
            continue

        print(f"\n=== Evaluating {name} ===")

        model = load_checkpoint(name, ckpt, num_classes)

        acc, loss = evaluate(model, val_loader)
        latency = measure_latency(model)
        size_mb = round(os.path.getsize(ckpt) / (1024 * 1024), 3)

        results[name] = {
            "checkpoint": ckpt,
            "accuracy": acc,
            "loss": loss,
            "latency_ms": latency,
            "size_mb": size_mb,
        }

        print(f"  Accuracy: {acc:.4f}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Latency: {latency:.2f} ms")
        print(f"  Size: {size_mb} MB")

    with open(os.path.join(REPORTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    if results:
        names = list(results.keys())
        plot_metric(names, [results[n]["accuracy"] for n in names], "Accuracy", "accuracy.png")
        plot_metric(names, [results[n]["latency_ms"] for n in names], "Latency (ms)", "latency.png")
        plot_metric(names, [results[n]["size_mb"] for n in names], "Model Size (MB)", "size.png")

        print("\nSaved evaluation results + plots.")
    else:
        print("\nNo valid checkpoints found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher", type=str, default=os.path.join(OUTPUT_DIR, "teacher_resnet50.pth"))
    parser.add_argument("--student", type=str, default=os.path.join(OUTPUT_DIR, "student_resnet18.pth"))
    parser.add_argument("--pruned", type=str, default=os.path.join(OUTPUT_DIR, "pruned_model.pth"))
    parser.add_argument("--distilled", type=str, default=os.path.join(OUTPUT_DIR, "distilled_model.pth"))
    parser.add_argument("--qat", type=str, default=os.path.join(OUTPUT_DIR, "quantized_model.pth"))

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))

    args = parser.parse_args()
    main(args)
