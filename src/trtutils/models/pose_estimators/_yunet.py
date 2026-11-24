# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from trtutils.image._pose_estimator import PoseEstimator
from trtutils.models._model import Model

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from trtutils.image._schema import PoseEstimatorOutputSchema


class YuNet(PoseEstimator, Model):
    """
    Alias of PoseEstimator with default args for the YuNet face detector.

    Each result is a face box and five landmarks: right eye, left eye, nose tip, right and
    left mouth corner. YuNet gives the landmarks no scores, so their third value repeats
    the face score. The OpenCV Zoo ONNX takes 0-255 pixels without mean / std, as OpenCV's
    ``FaceDetectorYN`` feeds it; the exporter flips the channels in the graph, so the model
    takes RGB like the other models. OpenCV runs YuNet on the image at its own size; the
    engine has a fixed size, so images are letterboxed to it, keeping the aspect ratio the
    faces were trained with. The thresholds are ``FaceDetectorYN``'s defaults, and the
    decode runs on the CPU.
    """

    _model_type = "yunet"
    _friendly_name = "YuNet"
    _default_imgsz = 640
    _imgsz_divisor: ClassVar[int | None] = 32
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 255.0),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.9,
        nms_iou_thres: float = 0.3,
        max_det: int = 300,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        output_schema: PoseEstimatorOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        postprocessor: str | None = None,
        cpu_resize: bool | None = None,
        antialias: bool = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        PoseEstimator.__init__(
            self,
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            conf_thres=conf_thres,
            nms_iou_thres=nms_iou_thres,
            max_det=max_det,
            mean=mean,
            std=std,
            output_schema=output_schema,
            dla_core=dla_core,
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            postprocessor=postprocessor,
            cpu_resize=cpu_resize,
            antialias=antialias,
            no_warn=no_warn,
            verbose=verbose,
        )
