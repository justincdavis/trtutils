# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision
import onnx
import onnxslim


def main(args: argparse.Namespace) -> None:
    """
    Export a torchvision classifier to ONNX.
    
    Parameters
    ----------
    args : argparse.Namespace
        The arguments to the script.

    """
    try:
        model_type = getattr(torchvision.models, args.model)
    except AttributeError:
        err_msg = f"Model {args.model} not found in torchvision.models"
        raise ValueError(err_msg)

    model = model_type(weights="DEFAULT")
    model.eval()

    dummy_input = torch.randn(args.batch_size, 3, args.imgsz, args.imgsz)
    torch.onnx.export(model, dummy_input, args.output)

    # always slim the model
    o_model = onnx.load(args.output)
    onnxslim.slim(o_model)
    onnx.save(o_model, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export a torchvision classifier to ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to export to onnx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="The path to the output ONNX file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="The size of the input images.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size of the input images.",
    )
    args = parser.parse_args()
    main(args)
