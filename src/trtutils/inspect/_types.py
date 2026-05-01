# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.compat._libs import trt


@dataclass
class LayerInfo:
    """
    Information about a single layer in a TensorRT network.

    Attributes
    ----------
    index : int
        The layer index in the network.
    name : str
        The name of the layer.
    layer_type : str
        The type of the layer (e.g., ``"CONVOLUTION"``, ``"POOLING"``).
    precision : trt.DataType
        The precision of the layer.
    input_tensor_size : int
        Total size of all input tensors in bytes.
    output_tensor_size : int
        Total size of all output tensors in bytes.
    dla_compatible : bool
        Whether the layer can run on a DLA accelerator.

    """

    index: int
    name: str
    layer_type: str
    precision: trt.DataType
    input_tensor_size: int
    output_tensor_size: int
    dla_compatible: bool

    def __str__(self: Self) -> str:
        dla_str = "DLA-compatible" if self.dla_compatible else "GPU-only"
        return (
            f"LayerInfo({self.index}: {self.name}, {self.layer_type}, {self.precision}, {dla_str})"
        )

    def __repr__(self: Self) -> str:
        return (
            f"LayerInfo(index={self.index}, name={self.name!r}, "
            f"layer_type={self.layer_type!r}, precision={self.precision!r}, "
            f"input_tensor_size={self.input_tensor_size}, "
            f"output_tensor_size={self.output_tensor_size}, "
            f"dla_compatible={self.dla_compatible})"
        )
