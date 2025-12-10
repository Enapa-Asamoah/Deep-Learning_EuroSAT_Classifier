import os
import sys
import json
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT

# Ensure project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.distillation_trainer import train_distillation
from src.training.pruning_trainer import train_pruning
from src.training.qat_trainer import train_qat

# ================================
# Directories
# ================================
RAW_DIR = 'data/raw'
SPLITS_DIR = 'data/splits'
OUTPUT_DIR = 'outputs/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Device
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================
# Dataset Loader
# ================================
def load_dataset(batch_size=32):
    # Use ImageNet normalization & 224 size so models and pretrained heads align
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)

    with open(os.path.join(SPLITS_DIR, 'train_val_test_split_seed42.json'), 'r') as f:
        splits = json.load(f)

    train_dataset = Subset(full_dataset, splits['train_indices'])
    val_dataset = Subset(full_dataset, splits['val_indices'])

    num_workers = min(8, (os.cpu_count() or 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(full_dataset.classes)

# ================================
# Load Model (robust)
# ================================
def load_model(model_type, checkpoint_path, num_classes, device):
    if model_type == 'teacher':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'student_resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'student_mobilenet_v3':
        model = models.mobilenet_v3_small(pretrained=True)
        # safe: replace last linear whatever its position is
        # classifier can be nn.Sequential([... , nn.Linear])
        if isinstance(model.classifier, torch.nn.Sequential):
            out_idx = -1
            model.classifier[out_idx] = torch.nn.Linear(model.classifier[out_idx].in_features, num_classes)
        else:
            # fallback
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    return model.to(device)

# ================================
# MAIN
# ================================
def main(args):
    # Load dataset
    train_loader, val_loader, num_classes = load_dataset(batch_size=args.batch_size)

    # Load teacher (if provided)
    teacher_model = None
    if args.teacher_checkpoint:
        teacher_model = load_model('teacher', args.teacher_checkpoint, num_classes, device)
        print(f"Loaded teacher checkpoint from {args.teacher_checkpoint}")

    # Load student
    student_model = load_model(args.student_arch, args.student_checkpoint, num_classes, device)
    print(f"Loaded student checkpoint from {args.student_checkpoint}")

    current_model = student_model

    # ============================================================
    # PRUNING
    # ============================================================
    if 'pruning' in args.methods:
        print("\n=== Running Pruning ===")
        pruned_model, pruning_history = train_pruning(
            current_model,                # prune the current model (student)
            train_loader, val_loader, device,
            epochs=args.epochs_dict.get('pruning', 10),
            lr=args.lr_dict.get('pruning', 1e-5),
            save_path=os.path.join(OUTPUT_DIR, 'pruned_model.pth')
        )
        with open(os.path.join(OUTPUT_DIR, 'pruned_history.json'), 'w') as f:
            json.dump(pruning_history, f)
        print("Pruned model saved.")
        current_model = pruned_model
    else:
        print("Skipping pruning...")

    # ============================================================
    # DISTILLATION
    # ============================================================
    if 'distillation' in args.methods:
        if teacher_model is None:
            raise ValueError("Distillation selected but no teacher model checkpoint provided.")
        print("\n=== Running Distillation ===")
        distilled_model, distill_history = train_distillation(
            current_model,
            teacher_model,
            train_loader, val_loader, device,
            alpha=args.alpha,
            temperature=args.temperature,
            epochs=args.epochs_dict.get('distillation', 20),
            lr=args.lr_dict.get('distillation', 1e-4),
            save_path=os.path.join(OUTPUT_DIR, 'distilled_model.pth')
        )
        with open(os.path.join(OUTPUT_DIR, 'distillation_history.json'), 'w') as f:
            json.dump(distill_history, f)
        print("Distilled model saved.")
        current_model = distilled_model
    else:
        print("Skipping distillation...")

    # ============================================================
    # QUANTIZATION (QAT)
    # ============================================================
    if 'qat' in args.methods:
        print("\n=== Running Quantization-Aware Training (QAT) ===")
        quantized_model, qat_history = train_qat(
            current_model,
            train_loader, val_loader, device,
            epochs=args.epochs_dict.get('qat', 10),
            lr=args.lr_dict.get('qat', 1e-5),
            save_path=os.path.join(OUTPUT_DIR, 'quantized_model.pth')
        )
        with open(os.path.join(OUTPUT_DIR, 'qat_history.json'), 'w') as f:
            json.dump(qat_history, f)
        print("QAT model saved.")
        current_model = quantized_model
    else:
        print("Skipping QAT...")

    print("\n=== Completed Selected Compression Techniques ===")


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compress a trained student model using pruning, distillation, or QAT.")
    parser.add_argument('--student_arch', type=str, required=True,
                        choices=['student_resnet18', 'student_mobilenet_v3'],
                        help="Architecture of the student model.")
    parser.add_argument('--student_checkpoint', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)

    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['pruning', 'distillation', 'qat'],
        default=['pruning', 'distillation', 'qat'],
        help="Choose which techniques to run."
    )

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=5.0)

    parser.add_argument('--epochs_dict', type=json.loads,
                        default='{"pruning":10,"distillation":20,"qat":10}')
    parser.add_argument('--lr_dict', type=json.loads,
                        default='{"pruning":1e-5,"distillation":1e-4,"qat":1e-5}')

    args = parser.parse_args()
    main(args)
