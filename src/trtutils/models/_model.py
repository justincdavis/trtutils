# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Model mixin providing concrete build() and download() classmethods."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from trtutils.builder._build import build_engine
from trtutils.builder._onnx import read_onnx
from trtutils.builder.hooks import yolo_efficient_nms_hook
from trtutils.builder.onnx import embed_sst as embed_sst_graph
from trtutils.compat._libs import trt

from ._utils import download_model_internal

if TYPE_CHECKING:
    from collections.abc import Callable


def nms_build_hook(
    *,
    num_classes: int = 80,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    top_k: int = 100,
    **_: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Build hook that adds EfficientNMS to a YOLO network.

    Parameters
    ----------
    num_classes : int
        Number of classes for EfficientNMS.
    conf_threshold : float
        Confidence threshold for EfficientNMS.
    iou_threshold : float
        IoU threshold for EfficientNMS.
    top_k : int
        Maximum number of detections to keep.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing hooks list.

    """
    return {
        "hooks": [
            yolo_efficient_nms_hook(
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                top_k=top_k,
            ),
        ],
    }


def depthwise_fp32_build_hook(*, onnx: Path | str, **_: Any) -> dict[str, Any]:  # noqa: ANN401
    """
    Build hook that forces FP32 for the depthwise convolutions.

    In FP16 the builder sometimes picks depthwise kernels that break the output (seen on
    TensorRT 10.15, RTX 5080): YOLOv10 loses large objects in about one build in seven,
    YOLOv13 detects nothing in about one in fifty. Depthwise convolutions are cheap, so
    FP32 costs little.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model.

    Returns
    -------
    dict[str, Any]
        Build engine overrides containing the layer_precision list.

    """
    network, _, _, _ = read_onnx(onnx)
    layer_precision = []
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        if layer.type != trt.LayerType.CONVOLUTION:
            continue
        layer.__class__ = trt.IConvolutionLayer
        if layer.num_groups > 1:
            layer_precision.append((idx, trt.DataType.FLOAT))
    return {"layer_precision": layer_precision or None, "prefer_precision_constraints": True}


def _embed_sst_scale(cls: type) -> float:
    """
    Scale implied by a detection wrapper's default ``input_range``.

    ``(0, 1)`` is ``1/255``. ``(0, 255)`` is ``1``. Wrappers that subtract a mean
    are a different preprocess and are rejected.

    Raises
    ------
    ValueError
        If the class normalizes with mean/std, or has no ``input_range``.

    """
    signature = inspect.signature(cls.__init__)
    mean = signature.parameters["mean"].default if "mean" in signature.parameters else None
    if mean is not None and mean is not inspect.Parameter.empty:
        err_msg = (
            f"{cls.__name__} normalizes with mean/std, which embed_sst does not cover. "
            "embed_sst is the scale, BGR swap, and HWC transpose used by detection models."
        )
        raise ValueError(err_msg)
    if "input_range" not in signature.parameters:
        err_msg = f"{cls.__name__} has no input_range, so embed_sst cannot choose a scale"
        raise ValueError(err_msg)
    low, high = signature.parameters["input_range"].default
    return (float(high) - float(low)) / 255.0


class Model:
    """
    Mixin providing concrete build() and download() classmethods driven by class attributes.

    Subclasses must define the following class attributes:

    Attributes
    ----------
    _model_type : str
        Download config key for the model family (e.g. ``"yolov11"``, ``"rfdetr"``).
    _friendly_name : str
        Human-readable name used in error messages (e.g. ``"YOLOv11"``, ``"RF-DETR"``).
    _default_imgsz : int
        Default input image size (e.g. ``640``, ``224``, ``518``).
    _input_tensors : list[tuple[str, str]]
        Input tensor definitions as ``(name, kind)`` pairs. ``kind`` is one of:

        - ``"image"`` — shape ``(batch, 3, imgsz, imgsz)``
        - ``"size"``  — shape ``(batch, 2)``

    _build_hooks : list[Callable[..., dict[str, Any]]]
        Optional. List of callables invoked during ``build()``. Each receives
        ``onnx``, ``imgsz``, ``batch_size``, plus any extra ``**kwargs`` from
        the caller, and returns a dict of ``build_engine()`` argument overrides.
        Defaults to ``[]``.
    _valid_imgszs : list[int] | None
        Optional. Exact set of allowed image sizes.
        Defaults to ``None`` (no restriction).
    _imgsz_divisor : int | None
        Optional. Image size must be divisible by this value.
        Defaults to ``None`` (no restriction).
    _model_imgszs : dict[str, int] | None
        Optional. Maps model-name substrings to expected image sizes for
        variant-dependent downloads (e.g. ``{"atto": 320, "femto": 416}``).
        Defaults to ``None``.
    _fp16 : bool
        Optional. Whether ``build()`` enables fp16. Defaults to ``True``.

    """

    _model_type: ClassVar[str]
    _friendly_name: ClassVar[str]
    _default_imgsz: ClassVar[int]
    _input_tensors: ClassVar[list[tuple[str, str]]]
    _build_hooks: ClassVar[list[Callable[..., dict[str, Any]]]] = []
    _valid_imgszs: ClassVar[list[int] | None] = None
    _imgsz_divisor: ClassVar[int | None] = None
    _model_imgszs: ClassVar[dict[str, int] | None] = None
    _fp16: ClassVar[bool] = True

    @classmethod
    def _validate_imgsz(cls, imgsz: int) -> None:
        if cls._valid_imgszs is not None and imgsz not in cls._valid_imgszs:
            err_msg = (
                f"{cls._friendly_name} supports only imgsz"
                f" of {', '.join(str(s) for s in cls._valid_imgszs)}, got {imgsz}"
            )
            raise ValueError(err_msg)
        if cls._imgsz_divisor is not None and imgsz % cls._imgsz_divisor != 0:
            err_msg = (
                f"{cls._friendly_name} supports only imgsz"
                f" divisible by {cls._imgsz_divisor}, got {imgsz}"
            )
            raise ValueError(err_msg)

    @classmethod
    def _make_shapes(
        cls,
        batch_size: int,
        imgsz: int,
    ) -> list[tuple[str, tuple[int, ...]]]:
        shapes: list[tuple[str, tuple[int, ...]]] = []
        for name, kind in cls._input_tensors:
            if kind == "image":
                shapes.append((name, (batch_size, 3, imgsz, imgsz)))
            elif kind == "size":
                shapes.append((name, (batch_size, 2)))
            else:
                err_msg = f"Unknown input tensor kind: {kind!r}"
                raise ValueError(err_msg)
        return shapes

    @classmethod
    def download(
        cls,
        model: str,
        output: Path | str,
        imgsz: int | None = None,
        opset: int = 17,
        *,
        simplify: bool = True,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download a model.

        Parameters
        ----------
        model : str
            The model to download.
        output : Path | str
            The output path to save the model to.
        imgsz : int, optional
            The image size to use for the model.
        opset : int
            The ONNX opset to use for the model. Default is 17.
        simplify : bool
            Whether to simplify the ONNX model after export. Default is True.
        no_cache : bool | None
            Disable caching of downloaded weights and repos.
        verbose : bool | None
            Print verbose output.

        Note
        ----
        Models downloaded through this method may have license restrictions.
        Users must ensure they comply with the model's license terms.

        """
        expected_imgsz = cls._default_imgsz
        if cls._model_imgszs is not None:
            for substring, size in cls._model_imgszs.items():
                if substring in model:
                    expected_imgsz = size
                    break

        if imgsz is None:
            imgsz = expected_imgsz
        elif cls._model_imgszs is not None and imgsz != expected_imgsz:
            err_msg = f"{cls._friendly_name} {model} requires imgsz of {expected_imgsz}, got {imgsz}"
            raise ValueError(err_msg)

        cls._validate_imgsz(imgsz)

        download_model_internal(
            model_type=cls._model_type,
            friendly_name=cls._friendly_name,
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            no_cache=no_cache,
            verbose=verbose,
        )

    @classmethod
    def build(
        cls,
        onnx: Path | str,
        output: Path | str,
        imgsz: int | None = None,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
        embed_sst: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Build a TensorRT engine.

        Parameters
        ----------
        onnx : Path | str
            Path to the ONNX model.
        output : Path | str
            Output path for the built engine.
        imgsz : int, optional
            Input image size used for shapes.
        batch_size : int
            Batch size for the engine. Default is 1.
        dla_core : int | None
            The DLA core to build the engine for.
        opt_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        verbose : bool | None
            Enable verbose builder output.
        embed_sst : bool
            Prepend scale, BGR-to-RGB, and HWC-to-CHW to the ONNX image input and
            build the engine on uint8 ``(batch, imgsz, imgsz, 3)``. Letterbox stays
            in the preprocessor. The scale comes from the wrapper's default
            ``input_range``: ``1/255`` for YOLO, ``1`` for YOLOX. Default is False.
        **kwargs : Any
            Additional keyword arguments. Consumed first by build hooks,
            then remaining kwargs are forwarded to ``build_engine()``.

        Raises
        ------
        ValueError
            If ``embed_sst`` is set on a model that normalizes with mean/std.

        """
        if imgsz is None:
            imgsz = cls._default_imgsz
        cls._validate_imgsz(imgsz)
        shapes = cls._make_shapes(batch_size, imgsz)
        onnx_path = Path(onnx)
        embedded: tempfile.TemporaryDirectory[str] | None = None
        if embed_sst:
            scale = _embed_sst_scale(cls)
            image_name = next(name for name, kind in cls._input_tensors if kind == "image")
            shapes = [
                ((name, (batch_size, imgsz, imgsz, 3)) if name == image_name else (name, shape))
                for name, shape in shapes
            ]
            embedded = tempfile.TemporaryDirectory()
            onnx_path = Path(embedded.name) / "embedded.onnx"
            embed_sst_graph(onnx, onnx_path, image_name, scale=scale, height=imgsz, width=imgsz)

        # Collect overrides from hooks, tracking consumed kwargs
        build_overrides: dict[str, Any] = {}
        consumed_keys: set[str] = set()
        for hook in cls._build_hooks:
            overrides = hook(onnx=onnx_path, imgsz=imgsz, batch_size=batch_size, **kwargs)
            consumed_keys.update(k for k in kwargs if k in inspect.signature(hook).parameters)
            for key, val in overrides.items():
                if key in build_overrides and isinstance(build_overrides[key], list):
                    build_overrides[key].extend(
                        val if isinstance(val, list) else [val],
                    )
                else:
                    build_overrides[key] = val

        # Forward unconsumed kwargs to build_engine for validation
        remaining_kwargs = {k: v for k, v in kwargs.items() if k not in consumed_keys}
        # an explicit fp16 wins; a strongly typed network takes its precision from the onnx
        fp16 = remaining_kwargs.pop(
            "fp16", cls._fp16 and not remaining_kwargs.get("strongly_typed", False)
        )

        try:
            build_engine(
                onnx=onnx_path,
                output=output,
                shapes=shapes,
                fp16=fp16,
                dla_core=dla_core,
                optimization_level=opt_level,
                verbose=verbose,
                **build_overrides,
                **remaining_kwargs,
            )
        finally:
            if embedded is not None:
                embedded.cleanup()
