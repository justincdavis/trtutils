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
from trtutils._flags import FLAGS
from trtutils.builder._build import build_engine
from trtutils.download._download import load_model_configs
from trtutils.download._tools import normalize_imgsz

from ._utils import download_model_internal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from trtutils.builder._build import BindingShape


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


def _hw(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    """Return ``(height, width)`` of an int (square) or ``(height, width)`` size."""
    return (imgsz, imgsz) if isinstance(imgsz, int) else imgsz


class Model:
    """
    Mixin providing concrete build() and download() classmethods driven by class attributes.

    ``download()`` also reads the model's download config: an ``imgsz`` entry
    is its default size and a ``patch`` entry the divisor its size must have.

    Image sizes are an int for a square input or a ``(height, width)`` tuple, in the
    ONNX ``(batch, 3, height, width)`` order. Note that ``ImageModel.input_shape``
    reports ``(width, height)`` instead.

    Subclasses must define the following class attributes:

    Attributes
    ----------
    _model_type : str
        Download config key for the model family (e.g. ``"yolov11"``, ``"rfdetr"``).
    _friendly_name : str
        Human-readable name used in error messages (e.g. ``"YOLOv11"``, ``"RF-DETR"``).
    _default_imgsz : int | tuple[int, int]
        Default input image size: an int for square (e.g. ``640``, ``224``, ``518``),
        or ``(height, width)`` for non-square (e.g. ``(256, 128)``).
    _input_tensors : list[tuple[str, str]]
        Input tensor definitions as ``(name, kind)`` pairs. ``kind`` is one of:

        - ``"image"`` — shape ``(batch, 3, height, width)``
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
        Optional. Image height and width must be divisible by this value.
        Defaults to ``None`` (no restriction).
    _model_imgszs : dict[str, int] | None
        Optional. Maps model-name substrings to expected image sizes for
        variant-dependent downloads (e.g. ``{"atto": 320, "femto": 416}``).
        Defaults to ``None``.
    _fp16 : bool
        Optional. Whether ``build()`` enables fp16. Defaults to ``True``.
    _dynamic_batch : bool
        Optional. Whether ``build()`` makes the batch dimension dynamic by default, so
        the engine runs any batch from 1 to ``batch_size``. Needs an ONNX with a
        symbolic batch axis. Defaults to ``False``.

    """

    _model_type: ClassVar[str]
    _friendly_name: ClassVar[str]
    _default_imgsz: ClassVar[int | tuple[int, int]]
    _input_tensors: ClassVar[list[tuple[str, str]]]
    _build_hooks: ClassVar[list[Callable[..., dict[str, Any]]]] = []
    _valid_imgszs: ClassVar[list[int] | None] = None
    _imgsz_divisor: ClassVar[int | None] = None
    _model_imgszs: ClassVar[dict[str, int] | None] = None
    _fp16: ClassVar[bool] = True
    _dynamic_batch: ClassVar[bool] = False

    @classmethod
    def _validate_imgsz(cls, imgsz: int | tuple[int, int]) -> None:
        if cls._valid_imgszs is not None and imgsz not in cls._valid_imgszs:
            err_msg = (
                f"{cls._friendly_name} supports only imgsz"
                f" of {', '.join(str(s) for s in cls._valid_imgszs)}, got {imgsz}"
            )
            raise ValueError(err_msg)
        if cls._imgsz_divisor is not None and any(s % cls._imgsz_divisor for s in _hw(imgsz)):
            err_msg = (
                f"{cls._friendly_name} supports only imgsz"
                f" divisible by {cls._imgsz_divisor}, got {imgsz}"
            )
            raise ValueError(err_msg)

    @classmethod
    def _onnx_imgsz(cls, onnx: Path | str) -> int | tuple[int, int] | None:
        """Read the static size of the ONNX image input, or None when unknown."""
        if not Path(onnx).is_file():
            return None
        image_name = next(name for name, kind in cls._input_tensors if kind == "image")
        # builder.onnx loads lazily: onnx is an optional extra
        try:
            hw = builder.onnx.get_static_hw(Path(onnx), image_name)
        except ImportError:
            return None
        return None if hw is None else normalize_imgsz(hw)

    @classmethod
    def _make_shapes(
        cls,
        batch_size: int,
        imgsz: int | tuple[int, int],
    ) -> list[tuple[str, tuple[int, ...]]]:
        height, width = _hw(imgsz)
        shapes: list[tuple[str, tuple[int, ...]]] = []
        for name, kind in cls._input_tensors:
            if kind == "image":
                shapes.append((name, (batch_size, 3, height, width)))
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
        imgsz: int | tuple[int, int] | None = None,
        opset: int = 17,
        *,
        simplify: Sequence[str] | bool | None = True,
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
        imgsz : int | tuple[int, int], optional
            The image size to use for the model: an int for square, or ``(height, width)``
            (the exporter must support non-square sizes). Defaults to the model's size.
        opset : int
            The ONNX opset to use for the model. Default is 17.
        simplify : Sequence[str] or bool, optional
            Whether and how to simplify the ONNX model after export. True (default) runs
            the default tools (polygraphy, onnxslim), a sequence of tool names runs those
            in order (valid: "polygraphy", "onnxslim", "onnxsim"), False or None skips it.
            Models over 2GB (weights in an external ``.data`` file) are not simplified.
        no_cache : bool | None
            Disable caching of downloaded weights and repos.
        verbose : bool | None
            Print verbose output.

        Raises
        ------
        ValueError
            If imgsz is invalid for the model, not divisible by its patch size, or
            non-square for an exporter that only takes square sizes.

        Notes
        -----
        Models downloaded through this method may have license restrictions.
        Users must ensure they comply with the model's license terms.

        """
        config = load_model_configs().get(cls._model_type, {}).get(model, {})
        expected_imgsz = normalize_imgsz(config.get("imgsz", cls._default_imgsz))
        if cls._model_imgszs is not None:
            for substring, size in cls._model_imgszs.items():
                if substring in model:
                    expected_imgsz = size
                    break

        imgsz = expected_imgsz if imgsz is None else normalize_imgsz(imgsz)
        if cls._model_imgszs is not None and imgsz != expected_imgsz:
            err_msg = f"{cls._friendly_name} {model} requires imgsz of {expected_imgsz}, got {imgsz}"
            raise ValueError(err_msg)

        cls._validate_imgsz(imgsz)
        patch = config.get("patch")
        if patch is not None and any(s % int(patch) for s in _hw(imgsz)):
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
        imgsz: int | tuple[int, int] | None = None,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
        embed_sst: bool = False,
        dynamic_batch: bool | None = None,
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
        imgsz : int | tuple[int, int], optional
            Input image size used for shapes: an int for square, or ``(height, width)``.
            This is the ONNX order; ``ImageModel.input_shape`` reports ``(width, height)``.
            When omitted, the static height and width of the ONNX image input are used,
            falling back to the class default for dynamic or non-NCHW inputs.
        batch_size : int
            Batch size for the engine, the maximum batch when ``dynamic_batch`` is on.
            Default is 1.
        dla_core : int | None
            The DLA core to build the engine for.
        opt_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        verbose : bool | None
            Enable verbose builder output.
        embed_sst : bool
            Prepend the BGR swap, the class's default ``input_range`` / ``mean`` /
            ``std`` normalization, and HWC-to-CHW to the ONNX image input, and build
            the engine on uint8 ``(batch, height, width, 3)``. Resize stays in the
            preprocessor. Default is False.
        dynamic_batch : bool, optional
            Build every input with a dynamic batch dimension: the engine runs any batch
            from 1 to ``batch_size`` and is tuned for ``batch_size``. The ONNX batch axis
            must be symbolic. Default is None, the class's ``_dynamic_batch``.
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
            imgsz = normalize_imgsz(cls._default_imgsz) if onnx_imgsz is None else onnx_imgsz
        else:
            imgsz = normalize_imgsz(imgsz)
            if onnx_imgsz is not None and imgsz != onnx_imgsz:
                err_msg = f"{cls._friendly_name} ONNX has a static size of {onnx_imgsz}, got {imgsz}"
                raise ValueError(err_msg)
        cls._validate_imgsz(imgsz)
        height, width = _hw(imgsz)
        shapes = cls._make_shapes(batch_size, imgsz)
        # the embedded copy lives until the engine is built (hooks read it too)
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(onnx)
            if embed_sst:
                input_range, mean, std = _default_norm(cls)
                image_name = next(name for name, kind in cls._input_tensors if kind == "image")
                shapes = [
                    ((name, (batch_size, height, width, 3)) if name == image_name else (name, shape))
                    for name, shape in shapes
                ]
                onnx_path = Path(tmp) / "embedded.onnx"
                # builder.onnx loads lazily: onnx is an optional extra
                builder.onnx.embed_sst(
                    onnx,
                    onnx_path,
                    image_name,
                    height=height,
                    width=width,
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
            # an explicit fp16 wins; a strongly typed network (always, on trt 11) takes
            # its precision from the onnx
            fp16 = remaining_kwargs.pop(
                "fp16",
                cls._fp16
                and FLAGS.WEAK_TYPING
                and not remaining_kwargs.get("strongly_typed", False),
            )

            # a dynamic batch runs 1..batch_size, tuned for batch_size
            dynamic_batch = dynamic_batch if dynamic_batch is not None else cls._dynamic_batch
            profiles: Sequence[tuple[str, BindingShape]] = shapes
            if dynamic_batch:
                profiles = [(name, ((1, *shape[1:]), shape, shape)) for name, shape in shapes]

            build_engine(
                onnx=onnx_path,
                output=output,
                shapes=profiles,
                fp16=fp16,
                dla_core=dla_core,
                optimization_level=opt_level,
                verbose=verbose,
                **build_overrides,
                **remaining_kwargs,
            )
