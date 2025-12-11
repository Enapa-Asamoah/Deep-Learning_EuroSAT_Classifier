"""Helpers to produce TorchScript models for PyTorch Mobile.

This module exports a small CLI to convert a PyTorch checkpoint to a
TorchScript file using tracing. The resulting `.pt` file can be further
optimized with `torch.utils.mobile_optimizer` if desired.

Usage:
python -m src.deployment.pytorch_mobile --arch resnet18 --checkpoint outputs/models/student_resnet18.pth --num-classes 10 --output outputs/models/student_resnet18_mobile.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


def _build_model(arch: str, num_classes: int) -> nn.Module:
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
    raise ValueError(f"Unsupported arch: {arch}")


def load_model(arch: str, checkpoint: str, num_classes: int, device: torch.device) -> nn.Module:
    model = _build_model(arch, num_classes)
    model.to(device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "weights"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt)
    model.eval()
    return model


def export_torchscript(model: nn.Module, output_path: str, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224), method: str = 'trace') -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    example = torch.randn(*input_shape)
    example = example.to(next(model.parameters()).device)

    if method == 'trace':
        scripted = torch.jit.trace(model, example)
    else:
        # try scripting as a fallback
        scripted = torch.jit.script(model)

    # Optionally optimize for mobile (requires newer torch)
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        scripted = optimize_for_mobile(scripted)
    except Exception:
        # mobile optimizer not available; continue
        pass

    scripted.save(output_path)
    print('Saved TorchScript model to', output_path)


def _cli():
    parser = argparse.ArgumentParser(description='Export checkpoint to TorchScript for mobile')
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input-size', type=int, nargs=3, default=[3, 224, 224])
    args = parser.parse_args()

    device = torch.device('cpu')
    model = load_model(args.arch, args.checkpoint, args.num_classes, device)
    input_shape = (1, args.input_size[0], args.input_size[1], args.input_size[2])
    export_torchscript(model, args.output, input_shape)


if __name__ == '__main__':
    _cli()
