"""Simple ONNX conversion utilities.

This module provides a small CLI and helper functions to export PyTorch models
to ONNX and run a basic validation check. It's intentionally lightweight so it
works in environments without every optional dependency. If `onnx` is
available the exported model will be validated with `onnx.checker`.

Usage (from repository root / lightvision):
python -m src.deployment.onnx_converter --arch resnet18 --checkpoint outputs/models/student_resnet18.pth --num-classes 10 --output outputs/models/student_resnet18.onnx
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

try:
    import onnx
    _HAS_ONNX = True
except Exception:
    _HAS_ONNX = False


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


def load_model_from_checkpoint(arch: str, checkpoint_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = _build_model(arch, num_classes)
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "weights"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt)
    model.eval()
    return model


def export_to_onnx(model: nn.Module, output_path: str, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224), opset: int = 13, dynamic_axes: bool = True) -> None:
    dummy = torch.randn(*input_shape)
    dummy = dummy.to(next(model.parameters()).device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_axes else None,
    )

    if _HAS_ONNX:
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model exported and validated:", output_path)
        except Exception as e:
            print("Exported ONNX file but validation failed:", e)
    else:
        print("ONNX exported but onnx package not installed - skipping validation.")


def _cli():
    parser = argparse.ArgumentParser(description="Export a PyTorch model checkpoint to ONNX")
    parser.add_argument("--arch", type=str, required=True, help="Architecture (resnet18,resnet50,mobilenet)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (state_dict or dict)")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 224, 224], help="C H W")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = load_model_from_checkpoint(args.arch, args.checkpoint, args.num_classes, device)
    input_shape = (1, args.input_size[0], args.input_size[1], args.input_size[2])
    export_to_onnx(model, args.output, input_shape, opset=args.opset)


if __name__ == "__main__":
    _cli()
