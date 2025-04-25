# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import torch
import onnx
import onnxsim
import trtutils


class Preproc(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    # swap BGR to RGB
    # divide all values by 255.0 (assume input is uint8, output is float16)
    # transpose from (H, W, C) to (1, C, H, W)
    def forward(self, image: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        image = image[..., [2, 1, 0]]  # BGR to RGB
        image = image[torch.newaxis, :]
        image = torch.permute(image, (0, 3, 1, 2))
        image = image.to(torch.float16)
        image = image * scale + offset
        return image


def main() -> None:
    # export to ONNX
    torch.onnx.export(
        Preproc(),
        (torch.zeros((640, 640, 3), dtype=torch.uint8), torch.zeros((1, 3, 640, 640), dtype=torch.float16), torch.ones((1, 3, 640, 640), dtype=torch.float16)),
        "preproc.onnx",
        input_names=["image", "offset", "scale"],
        output_names=["output"],
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
    )

    # simplify the ONNX model
    model = onnx.load("preproc.onnx")
    model_simplified, check = onnxsim.simplify(model)
    if check:
        onnx.save(model_simplified, "preproc_simplified.onnx")
        print("Simplified ONNX model saved.")
    else:
        print("Simplification failed.")
        onnx.save(model, "preproc_simplified.onnx")

    # build the tensorRT engine using trtexec
    trtutils.trtexec.run_trtexec(
        "--onnx=preproc_simplified.onnx --saveEngine=preproc.engine --fp16 --workspace=1024 --verbose"
    )


if __name__ == "__main__":
    main()
