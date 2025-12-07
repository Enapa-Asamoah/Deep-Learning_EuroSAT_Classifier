import os
import sys
import argparse
import json

# Ensure the project root (the folder containing `src`) is on sys.path so
# `from src...` imports work when running this script directly.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torchvision import models

from src.training.combined_trainer import train_combined
from src.training.distillation_trainer import train_distillation
from src.training.pruning_trainer import train_pruning
from src.training.qat_trainer import train_qat

# Directories
RAW_DIR = 'data/raw'
SPLITS_DIR = 'data/splits'
OUTPUT_DIR = 'outputs/models'
PLOTS_DIR = 'outputs/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_dataset_and_dataloaders(batch_size=32):
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import EuroSAT
    from torchvision import transforms
    # load transforms and dataset
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.344,0.380,0.408], std=[0.180,0.166,0.160])
    ])
    full_dataset = EuroSAT(root=RAW_DIR, download=True, transform=transform)

    with open(os.path.join(SPLITS_DIR, 'train_val_test_split_seed42.json'), 'r') as f:
        splits = json.load(f)

    train_dataset = Subset(full_dataset, splits['train_indices'])
    val_dataset = Subset(full_dataset, splits['val_indices'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, full_dataset.classes


def build_teacher(device, num_classes):
    teacher = models.resnet50(pretrained=True)
    in_features = teacher.fc.in_features
    teacher.fc = nn.Linear(in_features, num_classes)
    return teacher.to(device)


def build_student(name, device, num_classes):
    if name == 'mobilenet_v2':
        student = models.mobilenet_v2(pretrained=True)
        in_features = student.classifier[-1].in_features
        student.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == 'mobilenet_v3':
        student = models.mobilenet_v3_small(pretrained=True)
        if isinstance(student.classifier, nn.Sequential):
            in_features = student.classifier[-1].in_features
            student.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = student.classifier.in_features
            student.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError('Unsupported student architecture')
    return student.to(device)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, classes = load_dataset_and_dataloaders(batch_size=args.batch_size)

    # build models
    student = build_student(args.student_arch, device, len(classes))
    teacher = None
    if 'distillation' in args.steps:
        teacher = build_teacher(device, len(classes))
        # load pretrained teacher checkpoint if provided
        if args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint):
            teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
            print(f"Loaded teacher checkpoint from {args.teacher_checkpoint}")

    # Run combined compression pipeline (uses sequential steps)
    model, combined_history = train_combined(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        steps=args.steps,
        epochs_dict=args.epochs_dict,
        lr_dict=args.lr_dict,
        alpha=args.alpha,
        temperature=args.temperature,
        save_path=os.path.join(OUTPUT_DIR, f'compressed_{args.student_arch}.pth')
    )

    # Save final history
    history_path = os.path.join(OUTPUT_DIR, f'compressed_{args.student_arch}_history.json')
    with open(history_path, 'w') as f:
        json.dump(combined_history, f)
    print(f"Saved combined history to {history_path}")

    # Save final model
    model_path = os.path.join(OUTPUT_DIR, f'compressed_{args.student_arch}_final.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved final compressed model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combined compression pipeline')
    parser.add_argument('--student_arch', type=str, default='mobilenet_v2', choices=['mobilenet_v2','mobilenet_v3'])
    parser.add_argument('--steps', nargs='+', default=['pruning','distillation','qat'],
                        help='Sequence of steps to apply')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=5.0)
    # epochs and lr dicts are passed as JSON strings
    parser.add_argument('--epochs_dict', type=json.loads, default='{"pruning":10,"distillation":20,"qat":10}')
    parser.add_argument('--lr_dict', type=json.loads, default='{"pruning":1e-5,"distillation":1e-4,"qat":1e-5}')

    args = parser.parse_args()
    main(args)
