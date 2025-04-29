# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import torch
import onnx
import onnxsim
import tensorrt as trt


class PreprocBase(torch.nn.Module):
    def forward(self, image: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        # x: (H, W, 3) uint8, dynamic H/W
        x = image.to(torch.float16)
        x = x * scale + offset
        x = x.unsqueeze(0)  # (1, H, W, 3)
        x = x[:, :, :, [2, 1, 0]]  # swap BGR to RGB
        x = x.permute(0, 3, 1, 2)  # (1, 3, H, W)
        return x


def main() -> None:
    output_path = Path(__file__).parent.parent / "src/trtutils/impls/_onnx/preproc_base.onnx"
    output_path_str = str(output_path.resolve())

    # other preprocessing models will have to be defined manually
    # may want to bundle a CLI tool to compile preprocessing models (downside is will need torch)
    # export to ONNX
    dummy_image = torch.ones((640, 640, 3), dtype=torch.uint8)
    dummy_scale = torch.tensor([1.0], dtype=torch.float16)
    dummy_offset = torch.tensor([0.0], dtype=torch.float16)
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
    else:
        err_msg = "Could not simplify model!"
        raise RuntimeError(err_msg)


if __name__ == "__main__":
    main()
