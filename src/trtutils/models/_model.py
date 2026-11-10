# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Model mixin providing concrete build() and download() classmethods."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from trtutils import builder
from trtutils.builder._build import build_engine
from trtutils.download._download import load_model_configs

from ._utils import download_model_internal

if TYPE_CHECKING:
    from collections.abc import Callable


def _default_norm(
    cls: type,
) -> tuple[
    tuple[float, float], tuple[float, float, float] | None, tuple[float, float, float] | None
]:
    """Return the ``input_range``, ``mean``, and ``std`` defaults of ``cls.__init__``."""
    params = inspect.signature(cls.__init__).parameters

    def default(name: str, fallback: Any) -> Any:  # noqa: ANN401
        param = params.get(name)
        if param is None or param.default is inspect.Parameter.empty:
            return fallback
        return param.default

    return default("input_range", (0.0, 1.0)), default("mean", None), default("std", None)


class Model:
    """
    Mixin providing concrete build() and download() classmethods driven by class attributes.

    ``download()`` also reads the model's download config: an ``imgsz`` entry
    is its default size and a ``patch`` entry the divisor its size must have.

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
    def _onnx_imgsz(cls, onnx: Path | str) -> int | None:
        """Read the static square size of the ONNX image input, or None when unknown."""
        if not Path(onnx).is_file():
            return None
        image_name = next(name for name, kind in cls._input_tensors if kind == "image")
        # builder.onnx loads lazily: onnx is an optional extra
        try:
            return builder.onnx.get_static_imgsz(Path(onnx), image_name)
        except ImportError:
            return None

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

        Raises
        ------
        ValueError
            If imgsz is invalid for the model, or not divisible by its patch size.

        Notes
        -----
        Models downloaded through this method may have license restrictions.
        Users must ensure they comply with the model's license terms.

        """
        config = load_model_configs().get(cls._model_type, {}).get(model, {})
        expected_imgsz = int(config["imgsz"]) if "imgsz" in config else cls._default_imgsz
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
        patch = config.get("patch")
        if patch is not None and imgsz % int(patch) != 0:
            err_msg = (
                f"{cls._friendly_name} {model} requires imgsz divisible by {patch}, got {imgsz}"
            )
            raise ValueError(err_msg)

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
            Input image size used for shapes. When omitted, the static square size of the
            ONNX image input is used, falling back to the class default for dynamic,
            non-square, or non-NCHW inputs.
        batch_size : int
            Batch size for the engine. Default is 1.
        dla_core : int | None
            The DLA core to build the engine for.
        opt_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        verbose : bool | None
            Enable verbose builder output.
        embed_sst : bool
            Prepend the BGR swap, the class's default ``input_range`` / ``mean`` /
            ``std`` normalization, and HWC-to-CHW to the ONNX image input, and build
            the engine on uint8 ``(batch, imgsz, imgsz, 3)``. Resize stays in the
            preprocessor. Default is False.
        **kwargs : Any
            Additional keyword arguments. Consumed first by build hooks,
            then remaining kwargs are forwarded to ``build_engine()``.

        Raises
        ------
        ValueError
            If ``embed_sst`` is set and the class sets only one of mean and std, or if
            ``imgsz`` differs from the ONNX's static image size.

        """
        onnx_imgsz = cls._onnx_imgsz(onnx)
        if imgsz is None:
            imgsz = cls._default_imgsz if onnx_imgsz is None else onnx_imgsz
        elif onnx_imgsz is not None and imgsz != onnx_imgsz:
            err_msg = f"{cls._friendly_name} ONNX has a static size of {onnx_imgsz}, got {imgsz}"
            raise ValueError(err_msg)
        cls._validate_imgsz(imgsz)
        shapes = cls._make_shapes(batch_size, imgsz)
        # the embedded copy lives until the engine is built (hooks read it too)
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(onnx)
            if embed_sst:
                input_range, mean, std = _default_norm(cls)
                image_name = next(name for name, kind in cls._input_tensors if kind == "image")
                shapes = [
                    ((name, (batch_size, imgsz, imgsz, 3)) if name == image_name else (name, shape))
                    for name, shape in shapes
                ]
                onnx_path = Path(tmp) / "embedded.onnx"
                # builder.onnx loads lazily: onnx is an optional extra
                builder.onnx.embed_sst(
                    onnx,
                    onnx_path,
                    image_name,
                    height=imgsz,
                    width=imgsz,
                    input_range=input_range,
                    mean=mean,
                    std=std,
                )

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
