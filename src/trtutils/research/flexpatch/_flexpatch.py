# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
FlexPatch main system.

Classes
-------
FlexPatch
    Main FlexPatch system for efficient object detection.

"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ._aggregator import PatchAggregator
from ._no_recommender import NewObjectRecommender
from ._tf_recommender import TrackingFailureRecommender
from ._tracker import ObjectTracker

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import DetectorInterface


class FlexPatch:
    """FlexPatch system for efficient object detection on high-resolution video."""

    def __init__(
        self: Self,
        detector: DetectorInterface,
        frame_size: tuple[int, int],
        cluster_size: tuple[int, int] = (640, 360),
        cell_size: tuple[int, int] = (20, 22),
        max_age: int = 10,
        tf_ratio: float = 0.75,
        use_ratio_packing: bool = True,
        tf_model_path: Path | str | None = None,
    ) -> None:
        """
        Initialize FlexPatch system.

        Parameters
        ----------
        detector : DetectorInterface
            The object detector to use for patch-based detection.
        frame_size : tuple[int, int]
            Size of input frames as (width, height).
        cluster_size : tuple[int, int], optional
            Size of patch cluster as (width, height), by default (640, 360).
        cell_size : tuple[int, int], optional
            Cell size for new object detection as (width, height), by default (20, 22).
        max_age : int, optional
            Maximum age for tracked objects, by default 10.
        tf_ratio : float, optional
            Ratio of tracking-failure to new-object patches, by default 0.75.
        use_ratio_packing : bool, optional
            Whether to use ratio-based packing, by default True.
        tf_model_path : Path | str, optional
            Path to saved tracking-failure model (joblib format).
            If provided, loads model from disk.

        """
        self.detector = detector
        self.frame_size = frame_size
        self.cluster_size = cluster_size
        self.use_ratio_packing = use_ratio_packing
        self.tf_ratio = tf_ratio
        
        # Initialize components
        self.tracker = ObjectTracker(max_age=max_age)
        self.tf_recommender = TrackingFailureRecommender(model_path=tf_model_path)
        self.no_recommender = NewObjectRecommender(
            frame_size=frame_size,
            cell_size=cell_size,
        )
        self.aggregator = PatchAggregator(cluster_size=cluster_size)
        
        # State
        self.initialized = False
        self.frame_count = 0

    def set_tf_model(
        self: Self,
        model: object,
    ) -> None:
        """
        Set a trained model for tracking-failure recommendation.

        Parameters
        ----------
        model : DecisionTreeClassifier
            Trained sklearn DecisionTreeClassifier.

        """
        self.tf_recommender.model = model  # type: ignore[assignment]

    def process_frame(
        self: Self,
        frame: np.ndarray,
        *,
        verbose: bool = False,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Process a frame and return detections.

        Parameters
        ----------
        frame : np.ndarray
            The frame to process.
        verbose : bool, optional
            Whether to print verbose information, by default False.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            List of detections as (bbox, confidence, class_id).

        """
        self.frame_count += 1
        
        # First frame: run full detection
        if not self.initialized:
            if verbose:
                print("FlexPatch: Initializing with full-frame detection")
            
            detections = self.detector.end2end(frame)
            self.tracker.init(frame, detections)
            self.initialized = True
            
            # Reset new object recommender for detected regions
            for bbox, _, _ in detections:
                self.no_recommender.reset_region(bbox)
            
            return detections
        
        # Update tracker
        tracked_objects = self.tracker.update(frame)
        
        if verbose:
            print(f"FlexPatch: Tracking {len(tracked_objects)} objects")
        
        # Get patch recommendations
        tf_patches = self.tf_recommender.recommend(tracked_objects)
        
        exclusion_bboxes = [obj.bbox for obj in tracked_objects]
        no_patches = self.no_recommender.recommend(frame, exclusion_bboxes)
        
        if verbose:
            print(f"FlexPatch: {len(tf_patches)} TF patches, {len(no_patches)} NO patches")
        
        # Aggregate patches
        if self.use_ratio_packing:
            packed_patches = self.aggregator.pack_with_ratio(
                tf_patches,
                no_patches,
                tf_ratio=self.tf_ratio,
            )
        else:
            all_patches = tf_patches + no_patches
            packed_patches = self.aggregator.pack(all_patches)
        
        if verbose:
            print(f"FlexPatch: Packed {len(packed_patches)} patches")
        
        # Run detection on patch cluster
        if packed_patches:
            detections = self._run_patch_detection(frame, packed_patches)
            
            # Update tracker with new detections
            self.tracker.refresh(detections)
            
            # Reset refresh intervals for detected regions
            for bbox, _, _ in detections:
                self.no_recommender.reset_region(bbox)
        else:
            # No patches recommended, use tracked objects
            detections = [
                (obj.bbox, obj.confidence, 0)
                for obj in tracked_objects
            ]
        
        return detections

    def _run_patch_detection(
        self: Self,
        frame: np.ndarray,
        packed_patches: list,
    ) -> list[tuple[tuple[int, int, int, int], float, int]]:
        """
        Run detection on packed patches.

        Parameters
        ----------
        frame : np.ndarray
            The original frame.
        packed_patches : list[PatchInfo]
            List of packed patches with cluster positions.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], float, int]]
            Detections mapped back to original frame coordinates.

        """
        # Create patch cluster image
        cluster = np.zeros(
            (self.cluster_size[1], self.cluster_size[0], 3),
            dtype=np.uint8,
        )
        
        # Fill cluster with patches
        for patch in packed_patches:
            x, y, w, h = patch.bbox
            cx, cy = patch.cluster_pos
            
            # Extract patch from original frame
            x_clipped = max(0, min(x, frame.shape[1] - 1))
            y_clipped = max(0, min(y, frame.shape[0] - 1))
            w_clipped = min(w, frame.shape[1] - x_clipped)
            h_clipped = min(h, frame.shape[0] - y_clipped)
            
            if w_clipped <= 0 or h_clipped <= 0:
                continue
            
            patch_img = frame[y_clipped:y_clipped+h_clipped, x_clipped:x_clipped+w_clipped]
            
            # Place in cluster
            cx_end = min(cx + w_clipped, self.cluster_size[0])
            cy_end = min(cy + h_clipped, self.cluster_size[1])
            
            if cx < cx_end and cy < cy_end:
                cluster[cy:cy_end, cx:cx_end] = patch_img[:cy_end-cy, :cx_end-cx]
        
        # Run detection on cluster
        cluster_detections = self.detector.end2end(cluster)
        
        # Map detections back to original frame coordinates
        full_frame_detections = []
        for det_bbox, conf, cls in cluster_detections:
            dx, dy, dw, dh = det_bbox
            
            # Find which patch this detection belongs to
            for patch in packed_patches:
                cx, cy = patch.cluster_pos
                _, _, pw, ph = patch.bbox
                
                # Check if detection center is within patch cluster bounds
                det_center_x = dx + dw / 2
                det_center_y = dy + dh / 2
                
                if cx <= det_center_x < cx + pw and cy <= det_center_y < cy + ph:
                    # Map to original frame coordinates
                    orig_x, orig_y, _, _ = patch.bbox
                    
                    mapped_x = orig_x + (dx - cx)
                    mapped_y = orig_y + (dy - cy)
                    
                    mapped_bbox = (int(mapped_x), int(mapped_y), int(dw), int(dh))
                    full_frame_detections.append((mapped_bbox, conf, cls))
                    break
        
        return full_frame_detections

    def reset(self: Self) -> None:
        """Reset the FlexPatch system state."""
        self.tracker = ObjectTracker(max_age=self.tracker.max_age)
        self.no_recommender = NewObjectRecommender(
            frame_size=self.frame_size,
            cell_size=self.no_recommender.cell_size,
        )
        self.initialized = False
        self.frame_count = 0

    @staticmethod
    def generate_training_data(
        images: list[np.ndarray],
        ground_truth: list[list[tuple[tuple[int, int, int, int], float, int]]],
        detector: DetectorInterface,
        csv_path: Path | str,
        model_save_path: Path | str | None = None,
        max_age: int = 10,
        iou_threshold_high: float = 0.0,
        iou_threshold_medium: float = 0.5,
        *,
        train_model: bool = True,
        max_depth: int = 5,
        min_samples_split: int = 10,
        random_state: int = 42,
        verbose: bool = False,
    ) -> TrackingFailureRecommender | None:
        """
        Generate training data for tracking-failure model from annotated video.

        This method processes a sequence of images with ground truth annotations,
        tracks objects, computes tracking quality features, and generates a CSV
        file suitable for training the tracking-failure recommender model.

        Parameters
        ----------
        images : list[np.ndarray]
            List of frames to process.
        ground_truth : list[list[tuple[tuple[int, int, int, int], float, int]]]
            Ground truth detections for each frame.
            Format: [[(bbox, confidence, class_id), ...], ...]
            bbox is (x, y, w, h).
        detector : DetectorInterface
            Detector to use for initial detection on first frame.
        csv_path : Path | str
            Path where the CSV training data will be saved.
        model_save_path : Path | str, optional
            Path to save the trained model (joblib format).
            If None and train_model=True, model is not saved to disk.
        max_age : int, optional
            Maximum age for tracked objects, by default 10.
        iou_threshold_high : float, optional
            IoU threshold for "high" priority (tracking failure).
            Values <= this threshold are labeled "high", by default 0.0.
        iou_threshold_medium : float, optional
            IoU threshold for "medium" priority.
            Values <= this threshold (but > high) are "medium", by default 0.5.
        train_model : bool, optional
            Whether to train and return a model after generating CSV, by default True.
        max_depth : int, optional
            Decision tree max depth if training, by default 5.
        min_samples_split : int, optional
            Decision tree min samples to split if training, by default 10.
        random_state : int, optional
            Random state for model training if training, by default 42.
        verbose : bool, optional
            Whether to print progress information, by default False.

        Returns
        -------
        TrackingFailureRecommender | None
            Trained recommender if train_model=True, else None.

        Raises
        ------
        ValueError
            If images and ground_truth have different lengths.

        Examples
        --------
        >>> images = [cv2.imread(f"frame_{i}.jpg") for i in range(100)]
        >>> ground_truth = load_annotations("annotations.json")
        >>> detector = YOLO("yolo.engine")
        >>> recommender = FlexPatch.generate_training_data(
        ...     images, ground_truth, detector, "training_data.csv"
        ... )
        >>> # Use the trained model
        >>> flexpatch = FlexPatch(detector, frame_size=(1920, 1080))
        >>> flexpatch.set_tf_model(recommender.model)

        Notes
        -----
        The generated CSV will have columns:
        - min_eig: Minimum eigenvalue of spatial gradient matrix
        - ncc: Normalized cross-correlation between frames
        - accel: Acceleration (velocity difference)
        - flow_std: Standard deviation of optical flow
        - confidence: Original detection confidence
        - iou_label: "high", "medium", or "low" based on IoU with ground truth

        """
        if len(images) != len(ground_truth):
            err_msg = f"Images and ground truth must have same length. "
            err_msg += f"Got {len(images)} images and {len(ground_truth)} ground truth."
            raise ValueError(err_msg)

        if len(images) < 2:
            err_msg = "Need at least 2 frames to generate training data."
            raise ValueError(err_msg)

        if verbose:
            print(f"Generating training data from {len(images)} frames...")

        # Initialize tracker
        tracker = ObjectTracker(max_age=max_age)

        # Run detection on first frame
        first_detections = detector.end2end(images[0])
        tracker.init(images[0], first_detections)

        if verbose:
            print(f"Initialized tracker with {len(first_detections)} detections")

        # Collect training samples
        training_samples = []

        # Process subsequent frames
        for frame_idx in range(1, len(images)):
            frame = images[frame_idx]
            gt_objects = ground_truth[frame_idx]

            # Update tracker
            tracked_objects = tracker.update(frame)

            if verbose and frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}/{len(images)}: {len(tracked_objects)} tracked objects")

            # For each tracked object, compute IoU with ground truth
            for obj in tracked_objects:
                # Find best matching ground truth
                best_iou = 0.0
                for gt_bbox, _, _ in gt_objects:
                    iou = tracker._compute_iou(obj.bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou

                # Determine label based on IoU
                if best_iou <= iou_threshold_high:
                    label = "high"
                elif best_iou <= iou_threshold_medium:
                    label = "medium"
                else:
                    label = "low"

                # Store sample
                sample = {
                    "min_eig": obj.min_eigenvalue,
                    "ncc": obj.ncc,
                    "accel": obj.acceleration,
                    "flow_std": obj.flow_std,
                    "confidence": obj.confidence,
                    "iou_label": label,
                }
                training_samples.append(sample)

            # Optionally refresh tracker with ground truth periodically
            # This simulates a real scenario where detection runs periodically
            if frame_idx % 5 == 0:  # Refresh every 5 frames
                tracker.refresh(gt_objects)

        if verbose:
            print(f"Collected {len(training_samples)} training samples")

        # Write to CSV
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="") as csvfile:
            fieldnames = ["min_eig", "ncc", "accel", "flow_std", "confidence", "iou_label"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_samples)

        if verbose:
            print(f"Saved training data to {csv_path}")

            # Print label distribution
            label_counts = {"high": 0, "medium": 0, "low": 0}
            for sample in training_samples:
                label_counts[sample["iou_label"]] += 1
            print(f"Label distribution: {label_counts}")

        # Train model if requested
        if train_model:
            if verbose:
                print("Training model...")

            recommender = TrackingFailureRecommender.train_from_csv(
                csv_path=csv_path,
                model_save_path=model_save_path,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
            )

            if verbose:
                if model_save_path:
                    print(f"✓ Model trained and saved to {model_save_path}")
                else:
                    print("✓ Model trained successfully")

            return recommender

        return None

    # =========================================================================
    # DetectorInterface Compatibility Methods
    # =========================================================================

    @property
    def engine(self: Self):
        """Get the underlying TRTEngine of the wrapped detector."""
        return self.detector.engine

    @property
    def name(self: Self) -> str:
        """Get the name of the FlexPatch system."""
        return f"FlexPatch({self.detector.name})"

    @property
    def input_shape(self: Self) -> tuple[int, int]:
        """Get the input shape (frame size)."""
        return self.frame_size

    @property
    def dtype(self: Self):
        """Get the dtype required by the model."""
        return self.detector.dtype

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
        Preprocess the input using the wrapped detector.

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
        return self.detector.preprocess(
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
        Postprocess the outputs using the wrapped detector.

        Note: FlexPatch typically handles postprocessing internally.
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
        return self.detector.postprocess(
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
        Run FlexPatch inference on input image.

        Note: This returns raw detections. For full FlexPatch behavior with
        tracking, use process_frame() instead.

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
            Whether to postprocess outputs (always True for FlexPatch)
        no_copy : bool, optional
            If True, outputs will not be copied (ignored)
        verbose : bool, optional
            Whether to log additional information

        Returns
        -------
        list[np.ndarray]
            The detections (not in standard detector format, use get_detections)
        """
        detections = self.process_frame(image, verbose=verbose or False)
        
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

        Note: If outputs is None, returns empty list.
        For proper FlexPatch behavior, use end2end() or process_frame() directly.

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
        Perform end-to-end FlexPatch inference.

        This is the recommended way to use FlexPatch.

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
        return self.process_frame(image, verbose=verbose or False)

