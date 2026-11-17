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


def create_network(
    workspace: float = 4.0,
    *,
    strongly_typed: bool = False,
) -> tuple[trt.INetworkDefinition, trt.IBuilder, trt.IBuilderConfig]:
    """
    Create an empty explicit-batch TensorRT network with its builder and config.

    Parameters
    ----------
    workspace : float
        The size of the workspace in gigabytes.
        Default is 4.0 GiB.
    strongly_typed : bool, optional
        If True, create the network with the STRONGLY_TYPED flag.
        By default, False.

    Returns
    -------
    tuple[trt.INetworkDefinition, trt.IBuilder, trt.IBuilderConfig]
        The network, builder, and config.

    Raises
    ------
    RuntimeError
        If strongly_typed is True but the installed TensorRT does not support
        strongly-typed networks.

    """
    builder = trt.Builder(LOG)
    config = builder.create_builder_config()

    # setup the workspace size
    workspace_bytes = int(workspace * (1 << 30))
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_bytes
    else:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    # make network
    # trt 11 dropped the flag, networks are always explicit batch there
    network_flags = 0
    if FLAGS.EXPLICIT_BATCH_FLAG:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        if not FLAGS.STRONGLY_TYPED_SUPPORTED:
            err_msg = (
                "Installed TensorRT does not support strongly-typed networks "
                "(NetworkDefinitionCreationFlag.STRONGLY_TYPED not found)."
            )
            raise RuntimeError(err_msg)
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return builder.create_network(network_flags), builder, config


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

    External data files are resolved relative to the ONNX model path.

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
        A parse failure lists every TensorRT ONNX parser error (node,
        operator where TensorRT reports it, error code and description) and
        any other errors TensorRT logged while parsing.

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

    network, builder, config = create_network(workspace, strongly_typed=strongly_typed)

    # setup parser
    parser = trt.OnnxParser(network, LOG)
    model_bytes = onnx_path.read_bytes()
    # some failures (e.g. an LRN bias below 1e-5) are only explained in the log
    with LOG.capture_errors() as logged_errors:
        # the path lets the parser resolve external weight files (>2GB models)
        if FLAGS.ONNX_PARSE_PATH:
            parsed = parser.parse(model_bytes, str(onnx_path))
        else:
            parsed = parser.parse(model_bytes)
    # tensorrt 8.6 reports success on bytes that are not a model, leaving an empty network
    if not parsed or network.num_inputs == 0:
        err_msg = _parse_error_message(onnx_path, parser, logged_errors)
        raise RuntimeError(err_msg)

    return network, builder, config, parser


def _parse_error_message(
    onnx_path: Path,
    parser: trt.IOnnxParser,
    logged_errors: list[str],
) -> str:
    descriptions: list[str] = []
    for idx in range(parser.num_errors):
        error = parser.get_error(idx)
        where = f"node {error.node()}" if error.node() >= 0 else "model"
        # newer tensorrt also names the node and its operator, 8.6 only gives the index
        op = error.node_operator() if hasattr(error, "node_operator") else ""
        name = error.node_name() if hasattr(error, "node_name") else ""
        if op or name:
            where += f" ({op} '{name}')"
        code = str(error.code()).split(".")[-1]
        desc = " ".join(error.desc().split())
        descriptions.append(f"{where} in {error.func()}: {code}: {desc}")

    lines = [f"Cannot parse ONNX file: {onnx_path}"]
    if descriptions:
        lines.append(f"TensorRT ONNX parser errors ({len(descriptions)}):")
        lines.extend(f"  {desc}" for desc in descriptions)
    else:
        lines.append("TensorRT parsed it without errors into a network with no inputs.")
    # skip log lines the parser errors already repeat
    extra = [" ".join(msg.split()) for msg in logged_errors]
    extra = [msg for msg in extra if msg and not any(msg in desc for desc in descriptions)]
    if extra:
        lines.append("TensorRT errors logged while parsing:")
        lines.extend(f"  {msg}" for msg in extra)
    return "\n".join(lines)
