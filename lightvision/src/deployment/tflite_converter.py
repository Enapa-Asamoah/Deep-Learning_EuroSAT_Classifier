"""TFLite conversion helpers.

This module provides a best-effort converter that uses TensorFlow if
available. Direct PyTorch -> TFLite conversion isn't provided by default;
we try to convert via ONNX -> TensorFlow SavedModel when the optional
dependencies are present. If TensorFlow (and onnx/onnx-tf) are not
installed the CLI will explain next steps.

Usage example (best-effort):
python -m src.deployment.tflite_converter --onnx outputs/models/student_resnet18.onnx --output outputs/models/student_resnet18.tflite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Optional

try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:
    tf = None
    _HAS_TF = False

try:
    import onnx
    from onnx_tf.backend import prepare as onnx_to_tf
    _HAS_ONNX_TF = True
except Exception:
    onnx = None
    onnx_to_tf = None
    _HAS_ONNX_TF = False


def onnx_to_saved_model(onnx_path: str, saved_model_dir: str) -> None:
    if not _HAS_ONNX_TF:
        raise RuntimeError("onnx-tf is not installed; can't convert ONNX -> SavedModel")
    model = onnx.load(onnx_path)
    tf_rep = onnx_to_tf(model)
    tf_rep.export_graph(saved_model_dir)


def saved_model_to_tflite(saved_model_dir: str, tflite_out: str, quantize: bool = False) -> None:
    if not _HAS_TF:
        raise RuntimeError("TensorFlow is not installed; can't convert SavedModel -> TFLite")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_out, 'wb') as f:
        f.write(tflite_model)


def _cli():
    parser = argparse.ArgumentParser(description="Convert ONNX (or SavedModel) to TFLite")
    parser.add_argument("--onnx", type=str, help="ONNX model path (intermediate)")
    parser.add_argument("--saved-model", type=str, help="SavedModel dir (skip ONNX step)")
    parser.add_argument("--output", type=str, required=True, help="Output .tflite path")
    parser.add_argument("--quantize", action='store_true', help="Apply post-training quantization (if TF present)")
    args = parser.parse_args()

    tmp_saved = None

    if args.saved_model:
        saved_model_dir = args.saved_model
    elif args.onnx:
        if not os.path.exists(args.onnx):
            raise FileNotFoundError(args.onnx)
        tmp_saved = os.path.join(os.path.dirname(args.output), 'tmp_saved_model')
        if os.path.exists(tmp_saved):
            shutil.rmtree(tmp_saved)
        print('Converting ONNX -> SavedModel (requires onnx-tf)')
        onnx_to_saved_model(args.onnx, tmp_saved)
        saved_model_dir = tmp_saved
    else:
        raise ValueError('Provide --onnx or --saved-model')

    print('Converting SavedModel -> TFLite (requires tensorflow)')
    saved_model_to_tflite(saved_model_dir, args.output, quantize=args.quantize)

    if tmp_saved:
        try:
            shutil.rmtree(tmp_saved)
        except Exception:
            pass


if __name__ == '__main__':
    if not _HAS_TF or not _HAS_ONNX_TF:
        print('Warning: TensorFlow and/or onnx-tf are not installed. The converter needs these packages.')
        print('Install with: pip install tensorflow onnx onnx-tf')
    _cli()
