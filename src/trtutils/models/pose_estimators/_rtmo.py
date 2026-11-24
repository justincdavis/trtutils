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


class RTMO(PoseEstimator, Model):
    """
    Alias of PoseEstimator with default args for RTMO.

    The OpenMMLab ONNX takes 0-255 pixels without mean / std after a centred letterbox
    with gray (114) padding, as mmpose's ``BottomupResize`` does. The exporter flips the
    channels in the graph, so the model takes RGB like the other models. The engine runs
    NMS (IoU 0.65) without a score threshold and returns its best 50 rows; postprocessing
    filters them by ``conf_thres`` (0.1, mmpose's ``score_thr``) and applies
    ``nms_iou_thres`` when it is stricter. Postprocessing runs on the CPU: the
    outputs are a few kilobytes. CUDA graphs default to off: the in-graph NMS yields
    data-dependent shapes inside the engine, which TensorRT cannot capture.
    """

    _model_type = "rtmo"
    _friendly_name = "RTMO"
    _default_imgsz = 640
    _valid_imgszs: ClassVar[list[int] | None] = [640]
    _input_tensors: ClassVar[list[tuple[str, str]]] = [("input", "image")]

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 255.0),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.65,
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
        cuda_graph: bool | None = False,
        postprocessor: str | None = None,
        cpu_resize: bool | None = None,
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
            no_warn=no_warn,
            verbose=verbose,
        )
