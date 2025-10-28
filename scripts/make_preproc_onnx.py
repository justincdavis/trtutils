# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import torch
import onnx
import onnxsim


class PreprocBase(torch.nn.Module):
    def forward(self, image: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        # x: (H, W, 3) uint8, dynamic H/W
        x = image.to(torch.float16)
        # x = image * scale  # bug here, the should use the scaled version
        x = x * scale  # correction here
        x += offset
        x = x.unsqueeze(0)  # (1, H, W, 3)
        x = x[:, :, :, [2, 1, 0]]  # swap BGR to RGB
        x = x.permute(0, 3, 1, 2)  # (1, 3, H, W)
        return x


class PreprocImageNet(torch.nn.Module):
    def forward(self, image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # image: (H, W, 3) uint8, dynamic H/W
        # mean: (1, 3, 1, 1) float32
        # std: (1, 3, 1, 1) float32
        x = image.to(torch.float32)
        x = x / 255.0  # normalize to [0, 1]
        x = x.unsqueeze(0)  # (1, H, W, 3)
        x = x[:, :, :, [2, 1, 0]]  # swap BGR to RGB
        x = x.permute(0, 3, 1, 2)  # (1, 3, H, W)
        x = (x - mean) / std  # ImageNet normalization
        return x


def export_base_preproc() -> None:
    """Export the base preprocessing model."""
    output_path = Path(__file__).parent.parent / "src/trtutils/image/_onnx/image_preproc_base.onnx"
    output_path_str = str(output_path.resolve())

    print(f"Exporting base preprocessing model to {output_path_str}")

    # export to ONNX
    dummy_image = torch.ones((640, 640, 3), dtype=torch.uint8)
    dummy_scale = torch.tensor([1.0], dtype=torch.float32)
    dummy_offset = torch.tensor([0.0], dtype=torch.float32)
    torch.onnx.export(
        PreprocBase(),
        (dummy_image, dummy_scale, dummy_offset),
        output_path_str,
        input_names=["input", "scale", "offset"],
        output_names=["output"],
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "height", 1: "width"},
            "output": {2: "height", 3: "width"}
        }
    )

    # simplify the ONNX model
    model = onnx.load(output_path_str)
    model_simplified, check = onnxsim.simplify(
        model,
        check_n=5,
        test_input_shapes={
            "input": (640, 640, 3),
            "scale": (1,),
            "offset": (1,)
        }
    )
    if check:
        onnx.save(model_simplified, output_path_str)
        print("Base preprocessing model exported successfully!")
    else:
        err_msg = "Could not simplify base preprocessing model!"
        raise RuntimeError(err_msg)


def export_imagenet_preproc() -> None:
    """Export the ImageNet preprocessing model."""
    output_path = Path(__file__).parent.parent / "src/trtutils/image/_onnx/image_preproc_imagenet.onnx"
    output_path_str = str(output_path.resolve())

    print(f"Exporting ImageNet preprocessing model to {output_path_str}")

    # export to ONNX
    dummy_image = torch.ones((640, 640, 3), dtype=torch.uint8)
    dummy_mean = torch.ones((1, 3, 1, 1), dtype=torch.float32) * 0.5
    dummy_std = torch.ones((1, 3, 1, 1), dtype=torch.float32)
    torch.onnx.export(
        PreprocImageNet(),
        (dummy_image, dummy_mean, dummy_std),
        output_path_str,
        input_names=["input", "mean", "std"],
        output_names=["output"],
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "height", 1: "width"},
            "output": {2: "height", 3: "width"}
        }
    )

    # simplify the ONNX model
    model = onnx.load(output_path_str)
    model_simplified, check = onnxsim.simplify(
        model,
        check_n=5,
        test_input_shapes={
            "input": (640, 640, 3),
            "mean": (1, 3, 1, 1),
            "std": (1, 3, 1, 1)
        }
    )
    if check:
        onnx.save(model_simplified, output_path_str)
        print("ImageNet preprocessing model exported successfully!")
    else:
        err_msg = "Could not simplify ImageNet preprocessing model!"
        raise RuntimeError(err_msg)


def main() -> None:
    # Export both preprocessing models
    export_base_preproc()
    export_imagenet_preproc()


if __name__ == "__main__":
    main()
