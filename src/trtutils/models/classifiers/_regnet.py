# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.builder._build import build_engine
from trtutils.image._classifier import Classifier
from trtutils.models._utils import download_model_internal

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class RegNet(Classifier):
    """Alias of Classifier with default args for RegNet."""

    _default_imgsz = 224

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0, 1),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        Classifier.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
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
        Download a RegNet model.

        Parameters
        ----------
        model : str
            The model to download.
        output : Path | str
            The output path to save the model to.
        imgsz : int, optional
            The image size to use for the model. Default is 224.
        opset : int
            The ONNX opset to use for the model. Default is 17.
        accept : bool
            Whether to accept the license terms for the model.
        no_cache : bool | None
            Disable caching of downloaded weights and repos.
        verbose : bool | None
            Print verbose output.

        """
        if imgsz is None:
            imgsz = RegNet._default_imgsz
        download_model_internal(
            model_type="torchvision_classifier",
            friendly_name="RegNet",
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
        Build a TensorRT engine for RegNet.

        Parameters
        ----------
        onnx : Path | str
            Path to the ONNX model.
        output : Path | str
            Output path for the built engine.
        imgsz : int, optional
            Input image size used for shapes. Default is 224.
        batch_size : int
            Batch size for the engine. Default is 1.
        dla_core : int | None
            The DLA core to build the engine for.
        opt_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        verbose : bool | None
            Enable verbose builder output.

        """
        if imgsz is None:
            imgsz = RegNet._default_imgsz
        shapes = [("input", (batch_size, 3, imgsz, imgsz))]
        build_engine(
            onnx=onnx,
            output=output,
            shapes=shapes,
            fp16=True,
            dla_core=dla_core,
            optimization_level=opt_level,
            verbose=verbose,
        )
