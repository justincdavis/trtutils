# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.builder._build import build_engine
from trtutils.image._detector import Detector
from trtutils.image._schema import InputSchema, OutputSchema
from trtutils.models._utils import download_model_internal

from ._archs import DETR

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class DEIM(DETR):
    """Alias of DETR with default args for DEIM."""

    _default_imgsz = 640

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        Detector.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
            input_schema=InputSchema.RT_DETR,
            output_schema=OutputSchema.DETR_LBS,
            dla_core=dla_core,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            no_warn=no_warn,
            verbose=verbose,
        )

    @staticmethod
    def download(
        model: str,
        output: Path | str,
        imgsz: int | None = None,
        opset: int = 17,
        *,
        accept: bool = False,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download a DEIM model.

        Parameters
        ----------
        model: str
            Model identifier to download.
        output: Path | str
            Output path to save the ONNX model.
        imgsz: int = 640
            Image size used for export.
        opset: int = 17
            ONNX opset to export with.
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.

        Raises
        ------
        ValueError
            If imgsz is not 640.

        """
        if imgsz is None:
            imgsz = DEIM._default_imgsz
        elif imgsz != DEIM._default_imgsz:
            err_msg = f"DEIM supports only an imgsz of {DEIM._default_imgsz}, got {imgsz}"
            raise ValueError(err_msg)
        download_model_internal(
            model_type="deim",
            friendly_name="DEIM",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            accept=accept,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int | None = None,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Build a TensorRT engine for DEIM.

        Parameters
        ----------
        onnx: Path | str
            Path to the ONNX model.
        output: Path | str
            Output path for the built engine.
        imgsz: int
            Input image size used for shapes.
            Default is 640
        batch_size: int = 1
            Batch size for the engine.
        dla_core: int | None = None
            The DLA core to build the engine for.
            By default, None or build the engine for GPU.
        opt_level: int = 3
            TensorRT builder optimization level (0-5).
            Default is 3.
        verbose: bool | None = None
            Enable verbose builder output.

        Raises
        ------
        ValueError
            If imgsz is not 640.

        """
        if imgsz is None:
            imgsz = DEIM._default_imgsz
        elif imgsz != DEIM._default_imgsz:
            err_msg = f"DEIM supports only an imgsz of {DEIM._default_imgsz}, got {imgsz}"
            raise ValueError(err_msg)
        shapes = [
            ("images", (batch_size, 3, imgsz, imgsz)),
            ("orig_target_sizes", (batch_size, 2)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class DEIMv2(DETR):
    """Alias of DETR with default args for DEIMv2."""

    _default_imgsz = 640

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        Detector.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
            input_schema=InputSchema.RT_DETR,
            output_schema=OutputSchema.DETR_LBS,
            dla_core=dla_core,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            extra_nms=extra_nms,
            agnostic_nms=agnostic_nms,
            no_warn=no_warn,
            verbose=verbose,
        )

    @staticmethod
    def download(
        model: str,
        output: Path | str,
        imgsz: int | None = None,
        opset: int = 17,
        *,
        accept: bool = False,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download a DEIMv2 model.

        Parameters
        ----------
        model: str
            Model identifier to download.
        output: Path | str
            Output path to save the ONNX model.
        imgsz: int = 640
            Input image size used for shapes.
            Atto models require 320
            Femto models require 416
            All others (pico, n, s, m, l) require 640
        opset: int = 17
            ONNX opset to export with.
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.

        Raises
        ------
        ValueError
            If imgsz doesn't match the expected value for the model variant.

        """
        expected_imgsz = 640
        if "atto" in model:
            expected_imgsz = 320
        elif "femto" in model:
            expected_imgsz = 416

        if imgsz is None:
            imgsz = expected_imgsz
        elif imgsz != expected_imgsz:
            err_msg = f"DEIMv2 {model} requires imgsz of {expected_imgsz}, got {imgsz}"
            raise ValueError(err_msg)
        download_model_internal(
            model_type="deimv2",
            friendly_name="DEIMv2",
            model=model,
            output=output,
            imgsz=expected_imgsz,
            opset=opset,
            no_cache=no_cache,
            accept=accept,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int | None = None,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Build a TensorRT engine for DEIMv2.

        Parameters
        ----------
        onnx: Path | str
            Path to the ONNX model.
        output: Path | str
            Output path for the built engine.
        imgsz: int = 640
            Input image size used for shapes.
            Atto is 320, 320
            Femto is 416, 416
            All others (pico, n, s, m, l) are 640, 640
        batch_size: int = 1
            Batch size for the engine.
        dla_core: int | None = None
            The DLA core to build the engine for.
            By default, None or build the engine for GPU.
        opt_level: int = 3
            TensorRT builder optimization level (0-5).
            Default is 3.
        verbose: bool | None = None
            Enable verbose builder output.

        Raises
        ------
            ValueError: If imgsz is not 320, 416, or 640.

        """
        if imgsz is None:
            imgsz = DEIMv2._default_imgsz
        if imgsz not in (320, 416, 640):
            err_msg = f"DEIMv2 supports only imgsz of 320, 416, or 640, got {imgsz}"
            raise ValueError(err_msg)
        shapes = [
            ("images", (batch_size, 3, imgsz, imgsz)),
            ("orig_target_sizes", (batch_size, 2)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )
