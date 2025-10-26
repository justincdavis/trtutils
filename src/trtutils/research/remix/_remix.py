# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Main Remix system integration.

Classes
-------
RemixSystem
    Integrates adaptive partition and selective execution on TRT detectors.

"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from trtutils._log import LOG

from ._controller import PIDController, PlanController
from ._distribution import ObjectDistributionExtractor
from ._estimator import PerformanceEstimator
from ._executor import SelectiveExecutor
from ._planner import AdaptivePartitionPlanner
from ._profiler import NNProfiler

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface


class RemixSystem:
    """
    Remix system for adaptive high-resolution object detection.
    
    Implements DetectorInterface for compatibility with standard TRT detectors.
    """

    def __init__(
        self: Self,
        detectors: list[DetectorInterface],
        oracle: DetectorInterface,
        latency_budget: float,
        profile_path: Path | str | None = None,
        plans_path: Path | str | None = None,
        *,
        load_existing: bool = True,
    ) -> None:
        """
        Initialize Remix system.

        Parameters
        ----------
        detectors : list[DetectorInterface]
            Available detector models.
        oracle : DetectorInterface
            High-accuracy detector for profiling and distribution extraction.
        latency_budget : float
            Target latency budget in seconds.
        profile_path : Path | str, optional
            Path to save/load network profiles, by default None.
        plans_path : Path | str, optional
            Path to save/load partition plans, by default None.
        load_existing : bool, optional
            Whether to load existing profiles/plans if available, by default True.

        """
        self.detectors = detectors
        self.oracle = oracle
        self.latency_budget = latency_budget

        self.profile_path = Path(profile_path) if profile_path else None
        self.plans_path = Path(plans_path) if plans_path else None

        # Components (initialized later)
        self.profiler: NNProfiler | None = None
        self.profiles: dict[str, dict] = {}
        self.estimator: PerformanceEstimator | None = None
        self.planner: AdaptivePartitionPlanner | None = None
        self.plans: list = []
        self.executor: SelectiveExecutor | None = None
        self.controller: PlanController | None = None

        # Try to load existing data
        if load_existing:
            self._load_existing()

    def _load_existing(self: Self) -> None:
        """Load existing profiles and plans if available."""
        if self.profile_path and self.profile_path.exists():
            try:
                self.profiles = NNProfiler.load_profiles(self.profile_path)
                LOG.info("Loaded existing profiles")
            except Exception as e:
                LOG.warning(f"Failed to load profiles: {e}")

        if self.plans_path and self.plans_path.exists():
            try:
                self.plans = AdaptivePartitionPlanner.load_plans(self.plans_path)
                LOG.info("Loaded existing plans")
            except Exception as e:
                LOG.warning(f"Failed to load plans: {e}")

    def profile_networks(
        self: Self,
        coco_path: Path | str,
        num_latency_runs: int = 20,
        max_images: int | None = None,
        *,
        save: bool = True,
        verbose: bool = False,
    ) -> dict[str, dict]:
        """
        Profile all detector networks.

        Parameters
        ----------
        coco_path : Path | str
            Path to COCO dataset.
        num_latency_runs : int, optional
            Number of latency measurement runs, by default 20.
        max_images : int, optional
            Maximum images for accuracy evaluation, by default None.
        save : bool, optional
            Whether to save profiles to disk, by default True.
        verbose : bool, optional
            Whether to log detailed information, by default False.

        Returns
        -------
        dict[str, dict]
            Network profiles.

        """
        LOG.info("Starting network profiling")

        self.profiler = NNProfiler(self.detectors)
        self.profiles = self.profiler.profile(
            coco_path,
            num_latency_runs=num_latency_runs,
            max_images=max_images,
            verbose=verbose,
        )

        if save and self.profile_path:
            self.profiler.save_profiles(self.profile_path)

        LOG.info("Network profiling complete")
        return self.profiles

    def generate_plans(
        self: Self,
        view_shape: tuple[int, int],
        historical_frames: list[np.ndarray],
        max_plans: int = 10,
        *,
        save: bool = True,
        verbose: bool = False,
    ) -> list:
        """
        Generate partition plans.

        Parameters
        ----------
        view_shape : tuple[int, int]
            Image dimensions as (width, height).
        historical_frames : list[np.ndarray]
            Historical frames for distribution extraction.
        max_plans : int, optional
            Maximum number of plans to generate, by default 10.
        save : bool, optional
            Whether to save plans to disk, by default True.
        verbose : bool, optional
            Whether to log detailed information, by default False.

        Returns
        -------
        list[PartitionPlan]
            Generated partition plans.

        """
        if not self.profiles:
            err_msg = "Must profile networks before generating plans"
            raise RuntimeError(err_msg)

        LOG.info("Starting plan generation")

        # Extract object distribution
        extractor = ObjectDistributionExtractor(self.oracle)
        distribution = extractor.extract(historical_frames, verbose=verbose)

        # Initialize estimator and planner
        self.estimator = PerformanceEstimator(self.profiles)
        self.planner = AdaptivePartitionPlanner(
            self.detectors,
            self.profiles,
            self.estimator,
        )

        # Generate plans
        self.plans = self.planner.generate(
            view_shape,
            distribution,
            self.latency_budget,
            max_plans=max_plans,
            verbose=verbose,
        )

        if save and self.plans_path:
            self.planner.save_plans(self.plans, self.plans_path)

        LOG.info(f"Generated {len(self.plans)} partition plans")
        return self.plans

    def initialize_runtime(
        self: Self,
        kp: float = 0.6,
        ki: float = 0.3,
        kd: float = 0.1,
        *,
        verbose: bool = False,
    ) -> None:
        """
        Initialize runtime components (executor and controller).

        Parameters
        ----------
        kp : float, optional
            PID proportional gain, by default 0.6.
        ki : float, optional
            PID integral gain, by default 0.3.
        kd : float, optional
            PID derivative gain, by default 0.1.
        verbose : bool, optional
            Whether to log information, by default False.

        """
        if not self.plans:
            err_msg = "Must generate plans before initializing runtime"
            raise RuntimeError(err_msg)

        if verbose:
            LOG.info("Initializing runtime components")

        # Create detector mapping
        detector_map = {d.name: d for d in self.detectors}

        # Initialize executor
        self.executor = SelectiveExecutor(detector_map)

        # Initialize controller
        pid = PIDController(kp=kp, ki=ki, kd=kd)
        self.controller = PlanController(self.plans, pid, self.latency_budget)

        if verbose:
            LOG.info("Runtime initialization complete")

    def run_frame(
        self: Self,
        frame: np.ndarray,
        *,
        verbose: bool = False,
    ) -> tuple[list[tuple[tuple[int, int, int, int], float, int]], float]:
        """
        Run inference on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        tuple[list[tuple[tuple[int, int, int, int], float, int]], float]
            Detections and actual latency.

        """
        if self.executor is None or self.controller is None:
            err_msg = "Must initialize runtime before running frames"
            raise RuntimeError(err_msg)

        # Get current plan
        plan = self.controller.get_current_plan()

        # Execute
        start = time.perf_counter()
        detections = self.executor.execute(frame, plan, verbose=verbose)
        latency = time.perf_counter() - start

        # Adjust plan for next frame
        self.controller.adjust(latency, verbose=verbose)

        if verbose:
            LOG.debug(
                f"Frame: plan={plan.plan_id}, "
                f"detections={len(detections)}, "
                f"latency={latency*1000:.2f}ms",
            )

        return detections, latency

    def run_video(
        self: Self,
        video_path: Path | str,
        output_path: Path | str | None = None,
        max_frames: int | None = None,
        *,
        verbose: bool = False,
    ) -> dict[str, float]:
        """
        Run inference on video.

        Parameters
        ----------
        video_path : Path | str
            Input video path.
        output_path : Path | str, optional
            Output video path, by default None.
        max_frames : int, optional
            Maximum frames to process, by default None.
        verbose : bool, optional
            Whether to log information, by default False.

        Returns
        -------
        dict[str, float]
            Statistics (avg_latency, avg_detections, etc.).

        """
        if self.executor is None or self.controller is None:
            err_msg = "Must initialize runtime before running video"
            raise RuntimeError(err_msg)

        LOG.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            err_msg = f"Failed to open video: {video_path}"
            raise RuntimeError(err_msg)

        # Setup output writer if requested
        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        # Process frames
        frame_count = 0
        total_latency = 0.0
        total_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            detections, latency = self.run_frame(frame, verbose=verbose)

            total_latency += latency
            total_detections += len(detections)
            frame_count += 1

            if verbose and frame_count % 100 == 0:
                LOG.info(f"Processed {frame_count} frames")

            # Draw detections if saving output
            if writer:
                for (x1, y1, x2, y2), conf, cls in detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls}: {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()

        # Compute statistics
        stats = {
            "total_frames": frame_count,
            "avg_latency": total_latency / frame_count if frame_count > 0 else 0.0,
            "avg_detections": total_detections / frame_count if frame_count > 0 else 0.0,
            "total_time": total_latency,
        }

        LOG.info(
            f"Video processing complete: {frame_count} frames, "
            f"avg latency={stats['avg_latency']*1000:.2f}ms",
        )

        return stats

    def reset(self: Self) -> None:
        """Reset runtime state."""
        if self.executor:
            self.executor.reset()
        if self.controller:
            self.controller.reset()

    # =========================================================================
    # DetectorInterface Compatibility Methods
    # =========================================================================

    @property
    def engine(self: Self):
        """Get the underlying TRTEngine of the primary detector."""
        if not self.detectors:
            err_msg = "No detectors available"
            raise RuntimeError(err_msg)
        return self.detectors[0].engine

    @property
    def name(self: Self) -> str:
        """Get the name of the Remix system."""
        return "RemixSystem"

    @property
    def input_shape(self: Self) -> tuple[int, int]:
        """Get the input shape (uses primary detector's shape)."""
        if not self.detectors:
            err_msg = "No detectors available"
            raise RuntimeError(err_msg)
        return self.detectors[0].input_shape

    @property
    def dtype(self: Self):
        """Get the dtype required by the model."""
        if not self.detectors:
            err_msg = "No detectors available"
            raise RuntimeError(err_msg)
        return self.detectors[0].dtype

    def preprocess(
        self: Self,
        image: np.ndarray,
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        """
        Preprocess the input using the primary detector.

        Parameters
        ----------
        image : np.ndarray
            The image to preprocess
        resize : str, optional
            The method to resize the image with
        method : str, optional
            The underlying preprocessor to use
        no_copy : bool, optional
            If True, do not copy the data
        verbose : bool, optional
            Whether or not to log additional information

        Returns
        -------
        tuple[np.ndarray, tuple[float, float], tuple[float, float]]
            The preprocessed inputs, rescale ratios, and padding values
        """
        if not self.detectors:
            err_msg = "No detectors available"
            raise RuntimeError(err_msg)
        
        return self.detectors[0].preprocess(
            image,
            resize=resize,
            method=method,
            no_copy=no_copy,
            verbose=verbose,
        )

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: tuple[float, float],
        padding: tuple[float, float],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Postprocess the outputs using the primary detector.

        Note: Remix typically handles postprocessing internally during execution.
        This method is provided for interface compatibility.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs to postprocess
        ratios : tuple[float, float]
            The rescale ratios used during preprocessing
        padding : tuple[float, float]
            The padding values used during preprocessing
        conf_thres : float, optional
            The confidence threshold to filter detections by
        no_copy : bool, optional
            If True, do not copy the data
        verbose : bool, optional
            Whether or not to log additional information

        Returns
        -------
        list[np.ndarray]
            The postprocessed outputs
        """
        if not self.detectors:
            err_msg = "No detectors available"
            raise RuntimeError(err_msg)
        
        return self.detectors[0].postprocess(
            outputs,
            ratios,
            padding,
            conf_thres=conf_thres,
            no_copy=no_copy,
            verbose=verbose,
        )

    def run(
        self: Self,
        image: np.ndarray,
        ratios: tuple[float, float] | None = None,
        padding: tuple[float, float] | None = None,
        conf_thres: float | None = None,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run Remix inference on input image.

        Note: This returns raw detections. For full Remix behavior with
        latency tracking, use run_frame() instead.

        Parameters
        ----------
        image : np.ndarray
            The data to run the model on
        ratios : tuple[float, float], optional
            The ratios generated during preprocessing (ignored)
        padding : tuple[float, float], optional
            The padding values used during preprocessing (ignored)
        conf_thres : float, optional
            Optional confidence threshold (ignored)
        preprocessed : bool, optional
            Whether inputs have been preprocessed (ignored)
        postprocess : bool, optional
            Whether to postprocess outputs (always True for Remix)
        no_copy : bool, optional
            If True, outputs will not be copied (ignored)
        verbose : bool, optional
            Whether to log additional information

        Returns
        -------
        list[np.ndarray]
            The detections (not in standard detector format, use get_detections)
        """
        if self.executor is None or self.controller is None:
            err_msg = "Must initialize runtime before running inference"
            raise RuntimeError(err_msg)

        detections, _ = self.run_frame(image, verbose=verbose or False)
        
        # Return in a list for interface compatibility
        # Note: This is not the standard detector output format
        # Use get_detections() or end2end() for proper format
        return [np.array(detections, dtype=object)]

    def __call__(
        self: Self,
        image: np.ndarray,
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray]:
        """
        Run the model on input (interface compatibility).

        Parameters
        ----------
        image : np.ndarray
            The data to run the model on
        preprocessed : bool, optional
            Whether inputs have been preprocessed
        postprocess : bool, optional
            Whether to postprocess outputs
        no_copy : bool, optional
            If True, outputs will not be copied
        verbose : bool, optional
            Whether to log additional information

        Returns
        -------
        list[np.ndarray]
            The outputs of the model
        """
        return self.run(
            image,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

    def get_detections(
        self: Self,
        outputs: list[np.ndarray] | None = None,
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Get detections from outputs.

        Note: If outputs is None, returns last detections from run_frame.
        For proper Remix behavior, use end2end() or run_frame() directly.

        Parameters
        ----------
        outputs : list[np.ndarray], optional
            The outputs from run() (ignored if None)
        conf_thres : float, optional
            The confidence threshold (ignored)
        nms_iou_thres : float, optional
            The NMS IoU threshold (ignored)
        extra_nms : bool, optional
            Whether to apply extra NMS (ignored)
        agnostic_nms : bool, optional
            Whether to use class-agnostic NMS (ignored)
        verbose : bool, optional
            Whether to log information

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            List of detections as (bbox, confidence, class_id)
        """
        if outputs is not None and len(outputs) > 0:
            # Extract detections from outputs array
            detections_obj = outputs[0]
            if isinstance(detections_obj, np.ndarray) and detections_obj.dtype == object:
                # Convert back from object array
                return list(detections_obj)
            
        # Return empty list if no detections available
        return []

    def end2end(
        self: Self,
        image: np.ndarray,
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        extra_nms: bool | None = None,
        agnostic_nms: bool | None = None,
        verbose: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Perform end-to-end Remix inference.

        This is the recommended way to use Remix.

        Parameters
        ----------
        image : np.ndarray
            Input image
        conf_thres : float, optional
            Confidence threshold (ignored)
        nms_iou_thres : float, optional
            NMS IoU threshold (ignored)
        extra_nms : bool, optional
            Whether to apply extra NMS (ignored)
        agnostic_nms : bool, optional
            Whether to use class-agnostic NMS (ignored)
        verbose : bool, optional
            Whether to log information

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            List of detections as (bbox, confidence, class_id)
        """
        if self.executor is None or self.controller is None:
            err_msg = "Must initialize runtime before running inference"
            raise RuntimeError(err_msg)

        detections, _ = self.run_frame(image, verbose=verbose or False)
        return detections

