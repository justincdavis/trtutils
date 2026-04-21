# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from pathlib import Path

from trtutils._config import CONFIG
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.compat._libs import trt


def read_onnx(
    onnx: Path | str,
    workspace: float = 4.0,
    *,
    strongly_typed: bool = False,
) -> tuple[
    trt.INetworkDefinition,
    trt.IBuilder,
    trt.IBuilderConfig,
    trt.IOnnxParser,
]:
    """
    Open an ONNX model and generate TensorRT network, builder, config, and parser.

    Parameters
    ----------
    onnx : Path, str
        The path to the onnx model.
    workspace : float
        The size of the workspace in gigabytes.
        Default is 4.0 GiB.
    strongly_typed : bool, optional
        If True, create the network with the STRONGLY_TYPED flag so that
        precision is determined by the ONNX graph (Q/DQ nodes) rather than
        builder flags. Required on Blackwell (SM 10.0+) for mixed INT8+FP8.
        By default, False.

    Returns
    -------
    tuple[trt.INetworkDefinition, trt.IBuilder, trt.IBuilderConfig, trt.IOnnxParser]
        The network, builder, config, and parser.

    Raises
    ------
    FileNotFoundError
        If the onnx model does not exist
    IsADirectoryError
        If the onnx model path is a directory
    ValueError
        If the onnx model path does not have .onnx extension
    RuntimeError
        If the ONNX model cannot be parsed, or if strongly_typed is True
        but the installed TensorRT does not support strongly-typed networks.

    """
    # load libnvinfer plugins
    CONFIG.load_plugins()

    onnx_path = Path(onnx).resolve()
    if not onnx_path.exists():
        err_msg = f"Could not find ONNX model at: {onnx_path}"
        raise FileNotFoundError(err_msg)
    if onnx_path.is_dir():
        err_msg = f"Path given is a directory: {onnx_path}"
        raise IsADirectoryError(err_msg)
    if onnx_path.suffix != ".onnx":
        err_msg = "File does not have .onnx extension"
        raise ValueError(err_msg)

    builder = trt.Builder(LOG)
    config = builder.create_builder_config()

    # setup the workspace size
    workspace_bytes = int(workspace * (1 << 30))
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_bytes
    else:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    # make network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        if not FLAGS.STRONGLY_TYPED_SUPPORTED:
            err_msg = (
                "Installed TensorRT does not support strongly-typed networks "
                "(NetworkDefinitionCreationFlag.STRONGLY_TYPED not found)."
            )
            raise RuntimeError(err_msg)
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)

    # setup parser
    parser = trt.OnnxParser(network, LOG)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                LOG.error(parser.get_error(error))
            err_msg = "Cannot parse ONNX file"
            raise RuntimeError(err_msg)

    return network, builder, config, parser
