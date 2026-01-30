# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.builder._build import build_engine
from trtutils.image._detector import Detector

from ._utils import download_model_internal

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class YOLO(Detector):
    """Alias of Detector with default args for YOLO."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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


class YOLOX(YOLO):
    """Alias of Detector with default args for YOLOX."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 255),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOX model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolox",
            friendly_name="YOLOX",
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
        Build a TensorRT engine for YOLOX.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO3(YOLO):
    """Alias of Detector with default args for YOLOv3."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv3 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov3",
            friendly_name="YOLOv3",
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
        Build a TensorRT engine for YOLOv3.

        Parameters
        ----------
        onnx: Path | str
            Path to the ONNX model.
        output: Path | str
            Output path for the built engine.
        imgsz: int = 640
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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO5(YOLO):
    """Alias of Detector with default args for YOLOv5."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv5 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov5",
            friendly_name="YOLOv5",
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
        Build a TensorRT engine for YOLOv5.

        Parameters
        ----------
        onnx: Path | str
            Path to the ONNX model.
        output: Path | str
            Output path for the built engine.
        imgsz: int = 640
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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO7(YOLO):
    """Alias of Detector with default args for YOLO7."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv7 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov7",
            friendly_name="YOLOv7",
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
        Build a TensorRT engine for YOLOv7.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO8(YOLO):
    """Alias of Detector with default args for YOLO8."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv8 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov8",
            friendly_name="YOLOv8",
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
        Build a TensorRT engine for YOLOv8.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO9(YOLO):
    """Alias of Detector with default args for YOLO9."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv9 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov9",
            friendly_name="YOLOv9",
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
        Build a TensorRT engine for YOLOv9.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO10(YOLO):
    """Alias of Detector with default args for YOLO10."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv10 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov10",
            friendly_name="YOLOv10",
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
        Build a TensorRT engine for YOLOv10.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO11(YOLO):
    """Alias of Detector with default args for YOLO11."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv11 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov11",
            friendly_name="YOLOv11",
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
        Build a TensorRT engine for YOLOv11.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO12(YOLO):
    """Alias of Detector with default args for YOLO12."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv12 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov12",
            friendly_name="YOLOv12",
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
        Build a TensorRT engine for YOLOv12.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )


class YOLO13(YOLO):
    """Alias of Detector with default args for YOLO13."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        dla_core: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
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
        Download a YOLOv13 model.

        Parameters
        ----------
        model: str
            The model to download.
        output: Path | str
            The output path to save the model to.
        imgsz: int = 640
            The image size to use for the model.
        opset: int = 17
            The ONNX opset to use for the model.
        *,
        accept: bool, default False
            Whether to accept the license terms for the model.
        no_cache: bool | None = None,
            Disable caching of downloaded weights and repos.
        verbose: bool | None = None,
            Print verbose output.

        """
        download_model_internal(
            model_type="yolov13",
            friendly_name="YOLOv13",
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
        Build a TensorRT engine for YOLOv13.

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

        """
        shapes = [("images", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )
