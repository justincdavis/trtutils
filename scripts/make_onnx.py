# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import torch
import onnx
import onnxsim


class Simple(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> None:
        data = inputs * 0.5
        data = data + 1.0
        return data


def main() -> None:
    onnx_path = Path(__file__).parent.parent / "data" / "simple.onnx"
    torch.onnx.export(
        Simple(),
        torch.rand((160, 160)),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    onnx_model = onnx.load(onnx_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model_simp, check = onnxsim.simplify(
            onnx_model,
            check_n=5,
            perform_optimization=True,
        )    
    onnx.save(model_simp, onnx_path)


if __name__ == "__main__":
    main()
