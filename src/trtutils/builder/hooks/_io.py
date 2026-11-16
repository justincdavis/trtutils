# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from trtutils._flags import FLAGS
from trtutils.builder._utils import check_dynamic_range
from trtutils.compat._libs import trt


def fp16_io_hook(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
    """
    Make every float32 network input and output float16.

    The DLA reads and writes float16 natively, so float16 I/O can spare a DLA engine
    the GPU reformats at its boundary. Inputs and outputs of other dtypes are left as
    they are.

    Parameters
    ----------
    network : trt.INetworkDefinition
        The network to modify.

    Returns
    -------
    trt.INetworkDefinition
        The same network, with float16 I/O.

    Raises
    ------
    RuntimeError
        If the network is strongly typed (``strongly_typed=True``, and every network on
        TensorRT 11), where the I/O dtypes come from the ONNX graph.

    """
    _check_weakly_typed(network)
    tensors = [network.get_input(idx) for idx in range(network.num_inputs)]
    tensors += [network.get_output(idx) for idx in range(network.num_outputs)]
    for tensor in tensors:
        if tensor.dtype == trt.float32:
            tensor.dtype = trt.float16
    return network


def int8_input_hook(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
    """
    Make every float32 and float16 network input int8 at scale 1.

    TensorRT's INT8 scale is max(abs(min), abs(max)) / 127, so the dynamic range
    (-127, 127) gives scale 1 and every int8 value, -128 included, reaches the network
    unchanged. A uint8 image fed as ``(image ^ 0x80).view(np.int8)`` arrives as
    ``image - 128``: add the 128 back inside the network, where it folds into the
    normalization. Build with ``int8=True``; ``int8_ranges_hook`` sets the ranges of
    the remaining tensors without calibration data.

    Parameters
    ----------
    network : trt.INetworkDefinition
        The network to modify.

    Returns
    -------
    trt.INetworkDefinition
        The same network, with int8 inputs.

    Raises
    ------
    RuntimeError
        If the installed TensorRT has no per-tensor dynamic ranges (TensorRT 11), or the
        network is strongly typed (``strongly_typed=True``).

    """
    check_dynamic_range()
    _check_weakly_typed(network)
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        if tensor.dtype in (trt.float32, trt.float16):
            tensor.dtype = trt.int8
            tensor.set_dynamic_range(-127.0, 127.0)
    return network


def _check_weakly_typed(network: trt.INetworkDefinition) -> None:
    if not FLAGS.WEAK_TYPING or (
        FLAGS.STRONGLY_TYPED_SUPPORTED
        and network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    ):
        err_msg = (
            "Retyping network I/O needs a weakly-typed network. Weak typing was removed in "
            "TensorRT 11, and strongly_typed=True networks take their I/O dtypes from the "
            "ONNX graph: export the ONNX with the I/O dtypes you need instead."
        )
        raise RuntimeError(err_msg)
