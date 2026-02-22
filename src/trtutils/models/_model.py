# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Model mixin providing concrete build() and download() classmethods."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

from trtutils.builder._build import build_engine
from trtutils.builder.hooks import yolo_efficient_nms_hook

from ._utils import download_model_internal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def yolo_nms_build_hook(
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


class Model:
    """Mixin providing concrete build() and download() classmethods driven by class attributes."""

    _model_type: ClassVar[str]
    _friendly_name: ClassVar[str]
    _default_imgsz: ClassVar[int]

    # Build: list of (tensor_name, shape_kind) where shape_kind is:
    #   "image" -> (batch, 3, imgsz, imgsz)
    #   "size"  -> (batch, 2)
    _input_tensors: ClassVar[list[tuple[str, str]]]

    # Build hooks: list of callables returning build_engine override dicts
    _build_hooks: ClassVar[list[Callable[..., dict[str, Any]]]] = []

    # Validation: exact allowed sizes or divisibility constraint
    _valid_imgszs: ClassVar[list[int] | None] = None
    _imgsz_divisor: ClassVar[int | None] = None

    # Download: variant-dependent imgsz mapping (e.g. {"atto": 320, "femto": 416})
    _model_imgszs: ClassVar[dict[str, int] | None] = None

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
        accept: bool = False,
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
        accept : bool
            Whether to accept the license terms for the model.
        no_cache : bool | None
            Disable caching of downloaded weights and repos.
        verbose : bool | None
            Print verbose output.

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
            accept=accept,
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
        **kwargs : Any
            Additional keyword arguments forwarded to build hooks.

        """
        if imgsz is None:
            imgsz = cls._default_imgsz
        cls._validate_imgsz(imgsz)
        shapes = cls._make_shapes(batch_size, imgsz)

        # Collect overrides from hooks, tracking consumed kwargs
        build_overrides: dict[str, Any] = {}
        consumed_keys: set[str] = set()
        for hook in cls._build_hooks:
            overrides = hook(onnx=onnx, imgsz=imgsz, batch_size=batch_size, **kwargs)
            consumed_keys.update(k for k in kwargs if k in inspect.signature(hook).parameters)
            for key, val in overrides.items():
                if key in build_overrides and isinstance(build_overrides[key], list):
                    build_overrides[key].extend(
                        val if isinstance(val, list) else [val],
                    )
                else:
                    build_overrides[key] = val

        # Reject unknown kwargs (typo protection)
        unknown = set(kwargs) - consumed_keys
        if unknown:
            err_msg = f"{cls.__name__}.build() got unexpected keyword arguments: {unknown}"
            raise TypeError(err_msg)

        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
            **build_overrides,
        )
