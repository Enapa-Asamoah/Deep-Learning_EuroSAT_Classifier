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

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

    test_dataset = Subset(full_dataset, splits['test_indices'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'mps')
    )
    return test_loader, len(full_dataset.classes)

# ================================
# Model factory helpers
# ================================
def _build_model_from_arch(arch, num_classes):
    arch = arch.lower()

    if arch in {"resnet50", "teacher_resnet50", "resnet_50"}:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch in {"resnet18", "student_resnet18", "resnet_18"}:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch in {"mobilenet_v3_small", "mobilenet", "mobile"}:
        model = models.mobilenet_v3_small(pretrained=False)
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture hint: {arch}")


def _guess_arch_from_state_dict(state_dict):
    sample_key = "layer1.0.conv1.weight"
    weight = state_dict.get(sample_key)
    if weight is None:
        return None

    if weight.ndim == 4:
        kernel_size = weight.shape[-1]
        if kernel_size == 3:
            return "resnet18"
        if kernel_size == 1:
            return "resnet50"
    return None


def _architecture_candidates(model_name, state_dict_arch_hint=None):
    name = (model_name or "").lower()
    candidates = []

    def add_candidate(arch):
        if arch and arch not in candidates:
            candidates.append(arch)

    if "mobile" in name or "mbnet" in name:
        add_candidate("mobilenet_v3_small")

    if "resnet50" in name or "teacher" in name:
        add_candidate("resnet50")

    if any(term in name for term in ["resnet18", "student", "distill", "qat"]):
        add_candidate("resnet18")

    if "pruned" in name:
        add_candidate("resnet18")
        add_candidate("resnet50")

    if state_dict_arch_hint:
        add_candidate(state_dict_arch_hint)

    if not candidates:
        add_candidate("resnet18")

    return candidates


# ================================
# Robust checkpoint loader
# ================================
def load_checkpoint(model_name, ckpt_path, num_classes):
    raw_ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = raw_ckpt
    arch_hint = None

    if isinstance(raw_ckpt, dict):
        arch_hint = raw_ckpt.get("arch") or raw_ckpt.get("architecture") or raw_ckpt.get("model_arch")
        for key in ["model_state_dict", "state_dict", "weights"]:
            if key in raw_ckpt and isinstance(raw_ckpt[key], dict):
                state_dict = raw_ckpt[key]
                break

    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain a state_dict.")

    inferred_from_weights = _guess_arch_from_state_dict(state_dict)
    candidates = _architecture_candidates(model_name, state_dict_arch_hint=arch_hint)
    if inferred_from_weights and inferred_from_weights not in candidates:
        candidates.insert(0, inferred_from_weights)

    load_errors = []

    for arch in candidates:
        try:
            model = _build_model_from_arch(arch, num_classes).to(device)
        except ValueError as arch_err:
            load_errors.append((arch, str(arch_err)))
            continue

        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as state_err:
            load_errors.append((arch, str(state_err)))
            continue

        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        if arch_hint or inferred_from_weights:
            print(f"  Loaded using inferred architecture: {arch}")

        return model

    error_msg = "; ".join([f"{arch}: {err}" for arch, err in load_errors]) or "No architecture candidates were tested."
    raise RuntimeError(f"Failed to load checkpoint {ckpt_path} for {model_name}. Details: {error_msg}")

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
        if device.type == "mps":
            torch.mps.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()

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
