#!/usr/bin/env python3
# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="unused-ignore"
"""Export a ResNet18 classifier model to ONNX for testing."""

from __future__ import annotations

from pathlib import Path


def export_resnet18_onnx(output_path: Path | str) -> None:
    """
    Export ResNet18 to ONNX format.

    Parameters
    ----------
    output_path : Path or str
        The path where the ONNX model will be saved.

    """
    import torch  # noqa: PLC0415
    import torchvision.models as models  # type: ignore[import-untyped] # noqa: PLC0415

    # Create model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )

    print(f"Exported ResNet18 to {output_path}")


if __name__ == "__main__":
    # Default output path
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    ONNX_PATH = DATA_DIR / "onnx" / "resnet18.onnx"

    export_resnet18_onnx(ONNX_PATH)
