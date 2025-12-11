"""Orchestration script to convert checkpoints to multiple deployable formats.

This script wraps the individual converters in `src/deployment/` and provides
a single CLI to produce ONNX, TFLite and TorchScript artifacts.

Example:
python scripts/deployment/convert_models.py \
  --arch resnet18 \
  --checkpoint outputs/models/student_resnet18.pth \
  --num-classes 10 \
  --onnx outputs/models/student_resnet18.onnx \
  --torchscript outputs/models/student_resnet18_mobile.pt \
  --tflite outputs/models/student_resnet18.tflite
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _call_module(module_path: str, args: list[str]):
    cmd = [sys.executable, '-m', module_path] + args
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description='Convert a checkpoint to ONNX/TFLite/TorchScript')
    parser.add_argument('--arch', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--onnx', help='Output ONNX path')
    parser.add_argument('--torchscript', help='Output TorchScript path')
    parser.add_argument('--tflite', help='Output TFLite path')
    parser.add_argument('--quantize-tflite', action='store_true', help='Apply TF Lite quantization (requires TF)')
    args = parser.parse_args()

    if args.onnx:
        _call_module('src.deployment.onnx_converter', ['--arch', args.arch, '--checkpoint', args.checkpoint, '--num-classes', str(args.num_classes), '--output', args.onnx])

    if args.torchscript:
        _call_module('src.deployment.pytorch_mobile', ['--arch', args.arch, '--checkpoint', args.checkpoint, '--num-classes', str(args.num_classes), '--output', args.torchscript])

    if args.tflite:
        # prefer converting from ONNX if the user requested both; else try TorchScript path
        if args.onnx and os.path.exists(args.onnx):
            _call_module('src.deployment.tflite_converter', ['--onnx', args.onnx, '--output', args.tflite] + (['--quantize'] if args.quantize_tflite else []))
        else:
            print('TFLite conversion prefers ONNX input. Provide --onnx or pre-saved SavedModel for reliable conversion.')


if __name__ == '__main__':
    main()
