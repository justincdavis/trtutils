# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from trtutils.image._detector import Detector
from trtutils.builder._build import build_engine
from ._utils import download_model_internal

if TYPE_CHECKING:
    from typing_extensions import Self


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
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        dla_core: int | None = None,
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.
        """
        download_model_internal(
            model_type="rtdetrv1",
            friendly_name="RT-DETRv1",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int = 640,
        batch_size: int = 1,
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
        verbose: bool | None = None
            Enable verbose builder output.
        """
        shapes = [
            ("images", (batch_size, 3, imgsz, imgsz)),
            ("orig_target_sizes", (batch_size, 2)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.
        """
        download_model_internal(
            model_type="rtdetrv2",
            friendly_name="RT-DETRv2",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int = 640,
        batch_size: int = 1,
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
        verbose: bool | None = None
            Enable verbose builder output.
        """
        shapes = [
            ("image", (batch_size, 3, imgsz, imgsz)),
            ("orig_target_sizes", (batch_size, 2)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.
        """
        download_model_internal(
            model_type="rtdetrv3",
            friendly_name="RT-DETRv3",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int = 640,
        batch_size: int = 1,
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
        verbose: bool | None = None
            Enable verbose builder output.
        """
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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.
        """
        download_model_internal(
            model_type="dfine",
            friendly_name="D-FINE",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int = 640,
        batch_size: int = 1,
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
        verbose: bool | None = None
            Enable verbose builder output.
        """
        shapes = [
            ("images", (batch_size, 3, imgsz, imgsz)),
            ("orig_target_sizes", (batch_size, 2)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.
        """
        download_model_internal(
            model_type="deim",
            friendly_name="DEIM",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int = 640,
        batch_size: int = 1,
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
        verbose: bool | None = None
            Enable verbose builder output.
        """
        shapes = [
            ("images", (batch_size, 3, imgsz, imgsz)),
            ("orig_target_sizes", (batch_size, 2)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
            Image size used for export.
        opset: int = 17
            ONNX opset to export with.
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.

        """
        download_model_internal(
            model_type="deimv2",
            friendly_name="DEIMv2",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int = 640,
        batch_size: int = 1,
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
            All others are 640, 640
        batch_size: int = 1
            Batch size for the engine.
        verbose: bool | None = None
            Enable verbose builder output.

        """
        shapes = [
            ("images", (batch_size, 3, imgsz, imgsz)),
            ("orig_target_sizes", (batch_size, 2)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
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
        no_cache: bool | None = None
            Disable caching of downloads.
        verbose: bool | None = None
            Enable verbose logging.

        """
        download_model_internal(
            model_type="rfdetr",
            friendly_name="RF-DETR",
            model=model,
            output=output,
            imgsz=imgsz,
            opset=opset,
            no_cache=no_cache,
            verbose=verbose,
        )

    @staticmethod
    def build(
        onnx: Path | str,
        output: Path | str,
        imgsz: int = 576,
        batch_size: int = 1,
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
        verbose: bool | None = None
            Enable verbose builder output.

        """
        shapes = [
            ("input", (batch_size, 3, imgsz, imgsz)),
        ]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            verbose=verbose,
        )
