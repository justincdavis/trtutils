# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.builder._build import build_engine
from trtutils.compat._libs import trt
from trtutils.image._detector import Detector
from trtutils.inspect._onnx import inspect_onnx_layers

from ._utils import download_model_internal

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

DEFAULT_DETR_IMGSZ = 640


class DETR(Detector):
    """Alias of Detector with default args for DETR."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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


class RTDETRv1(DETR):
    """Alias of DETR with default args for RT-DETRv1."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 255),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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
        imgsz: int = 640,
        opset: int = 17,
        *,
        accept: bool = False,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download an RT-DETRv1 model.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"RT-DETRv1 supports only an imgsz of 640, got {imgsz}"
            raise ValueError(err_msg)
        download_model_internal(
            model_type="rtdetrv1",
            friendly_name="RT-DETRv1",
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
        imgsz: int = 640,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Build a TensorRT engine for RT-DETRv1.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"RT-DETRv1 supports only an imgsz of 640, got {imgsz}"
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


class RTDETRv2(DETR):
    """Alias of DETR with default args for RT-DETRv2."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 255),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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
        imgsz: int = 640,
        opset: int = 17,
        *,
        accept: bool = False,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download an RT-DETRv2 model.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"RT-DETRv2 supports only an imgsz of 640, got {imgsz}"
            raise ValueError(err_msg)
        download_model_internal(
            model_type="rtdetrv2",
            friendly_name="RT-DETRv2",
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
        imgsz: int = 640,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Build a TensorRT engine for RT-DETRv2.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"RT-DETRv2 supports only an imgsz of 640, got {imgsz}"
            raise ValueError(err_msg)
        shapes = [
            ("image", (batch_size, 3, imgsz, imgsz)),
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


class RTDETRv3(DETR):
    """Alias of DETR with default args for RT-DETRv3."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 255),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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
        imgsz: int = 640,
        opset: int = 17,
        *,
        accept: bool = False,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download an RT-DETRv3 model.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"RT-DETRv3 supports only an imgsz of 640, got {imgsz}"
            raise ValueError(err_msg)
        download_model_internal(
            model_type="rtdetrv3",
            friendly_name="RT-DETRv3",
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
        imgsz: int = 640,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Build a TensorRT engine for RT-DETRv3.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"RT-DETRv3 supports only an imgsz of 640, got {imgsz}"
            raise ValueError(err_msg)
        shapes = [
            ("image", (batch_size, 3, imgsz, imgsz)),
            ("im_shape", (batch_size, 2)),
            ("scale_factor", (batch_size, 2)),
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


class DFINE(DETR):
    """Alias of DETR with default args for D-FINE."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "linear",
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
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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
        imgsz: int = 640,
        opset: int = 17,
        *,
        accept: bool = False,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download a D-FINE model.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"D-FINE supports only an imgsz of 640, got {imgsz}"
            raise ValueError(err_msg)
        download_model_internal(
            model_type="dfine",
            friendly_name="D-FINE",
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
        imgsz: int = 640,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Build a TensorRT engine for D-FINE.

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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"D-FINE supports only an imgsz of 640, got {imgsz}"
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


class DEIM(DETR):
    """Alias of DETR with default args for DEIM."""

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
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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
        imgsz: int = 640,
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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"DEIM supports only an imgsz of 640, got {imgsz}"
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
        imgsz: int = 640,
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
        if imgsz != DEFAULT_DETR_IMGSZ:
            err_msg = f"DEIM supports only an imgsz of 640, got {imgsz}"
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
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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
        imgsz: int = 640,
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

        if imgsz != expected_imgsz:
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
        imgsz: int = 640,
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


class RFDETR(DETR):
    """Alias of DETR with default args for RF-DETR."""

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
        cuda_graph: bool | None = True,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            mean=mean,
            std=std,
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
        imgsz: int = 576,
        opset: int = 17,
        *,
        accept: bool = False,
        no_cache: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Download an RF-DETR model.

        Parameters
        ----------
        model: str
            Model identifier to download.
        output: Path | str
            Output path to save the ONNX model.
        imgsz: int = 576
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
            If imgsz is not divisible by 32.

        """
        if imgsz % 32 != 0:
            err_msg = f"RF-DETR supports only imgsz divisible by 32, got {imgsz}"
            raise ValueError(err_msg)
        download_model_internal(
            model_type="rfdetr",
            friendly_name="RF-DETR",
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
        imgsz: int = 576,
        batch_size: int = 1,
        dla_core: int | None = None,
        opt_level: int = 3,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Build a TensorRT engine for RF-DETR.

        Parameters
        ----------
        onnx: Path | str
            Path to the ONNX model.
        output: Path | str
            Output path for the built engine.
        imgsz: int
            Input image size used for shapes.
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
            ValueError: If imgsz is not divisible by 32.

        """
        if imgsz % 32 != 0:
            err_msg = f"RF-DETR supports only imgsz divisible by 32, got {imgsz}"
            raise ValueError(err_msg)
        shapes = [
            ("input", (batch_size, 3, imgsz, imgsz)),
        ]
        layer_info = inspect_onnx_layers(onnx, verbose=True)
        layer_precision: list[tuple[int, trt.DataType]] = []
        for idx, name, _, _ in layer_info:
            lower_name = name.lower()
            if "reducemean" in lower_name or "downsample" in lower_name:
                layer_precision.append((idx, trt.DataType.FLOAT))
            else:
                layer_precision.append((idx, trt.DataType.HALF))
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            layer_precision=layer_precision,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )
