# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Tracking-failure patch recommender.

Classes
-------
TrackingFailureRecommender
    Recommends patches where tracking likely failed using ML.

CSV Training Format
-------------------
The train_from_csv method expects a CSV file with the following columns:
- min_eig: float (minimum eigenvalue of spatial gradient matrix)
- ncc: float (normalized cross-correlation between frames, range [0, 1])
- accel: float (acceleration/velocity difference in pixels)
- flow_std: float (standard deviation of optical flow errors)
- confidence: float (detector confidence score, range [0, 1])
- iou_label: str ("high", "medium", or "low")
    - "high": IoU = 0 (complete tracking failure)
    - "medium": 0 < IoU ≤ 0.5 (partial tracking failure)
    - "low": IoU > 0.5 (tracking succeeded)

Example CSV:
```
min_eig,ncc,accel,flow_std,confidence,iou_label
0.002,0.85,2.3,1.2,0.9,low
0.001,0.45,15.7,8.5,0.7,high
0.003,0.72,5.2,3.1,0.8,medium
```

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier
    from typing_extensions import Self

    from ._tracker import TrackedObject

try:
    from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier
    import joblib as _joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    _DecisionTreeClassifier = None  # type: ignore[assignment, misc]
    _joblib = None  # type: ignore[assignment]


class TrackingFailureRecommender:
    """Recommends patches for tracking failure detection."""

    def __init__(
        self: Self,
        model: DecisionTreeClassifier | None = None,
        model_path: Path | str | None = None,
        padding_ratio: float = 1.0,
    ) -> None:
        """
        Initialize the tracking failure recommender.

        Parameters
        ----------
        model : DecisionTreeClassifier, optional
            Pre-trained decision tree model. If None, a default model
            with basic heuristics will be used.
        model_path : Path | str, optional
            Path to a saved model file (joblib format).
            If provided, loads model from disk.
        padding_ratio : float, optional
            Ratio of padding to add around tracked boxes.
            Padding = bbox_size * padding_ratio / 2 on each side.
            Default is 1.0 (padding equals box dimensions).

        Raises
        ------
        ImportError
            If sklearn/joblib not installed when model is provided.

        """
        if (model is not None or model_path is not None) and not SKLEARN_AVAILABLE:
            err_msg = "scikit-learn and joblib are required for ML-based recommender. "
            err_msg += "Install with: pip install scikit-learn joblib"
            raise ImportError(err_msg)
        
        if model_path is not None:
            self.model = self.load_model(model_path)
        else:
            self.model = model
        
        self.padding_ratio = padding_ratio

    @staticmethod
    def train_from_csv(
        csv_path: Path | str,
        model_save_path: Path | str | None = None,
        max_depth: int | None = 5,
        min_samples_split: int = 10,
        random_state: int = 42,
    ) -> TrackingFailureRecommender:
        """
        Train a decision tree model from CSV data.

        Parameters
        ----------
        csv_path : Path | str
            Path to CSV file with training data.
        model_save_path : Path | str, optional
            Path to save the trained model (joblib format).
            If None, model is not saved to disk.
        max_depth : int, optional
            Maximum depth of the decision tree, by default 5.
        min_samples_split : int, optional
            Minimum samples required to split a node, by default 10.
        random_state : int, optional
            Random state for reproducibility, by default 42.

        Returns
        -------
        TrackingFailureRecommender
            A new recommender with trained model.

        Raises
        ------
        ImportError
            If scikit-learn/joblib is not installed.

        """
        if not SKLEARN_AVAILABLE:
            err_msg = "scikit-learn and joblib are required for training. "
            err_msg += "Install with: pip install scikit-learn joblib"
            raise ImportError(err_msg)
        
        import pandas as pd
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ["min_eig", "ncc", "accel", "flow_std", "confidence", "iou_label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            err_msg = f"Missing required columns: {missing_cols}"
            raise ValueError(err_msg)
        
        # Prepare features and labels
        X = df[["min_eig", "ncc", "accel", "flow_std", "confidence"]].values
        y = df["iou_label"].values
        
        # Train decision tree
        model = _DecisionTreeClassifier(  # type: ignore[misc]
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        model.fit(X, y)
        
        # Save model if path provided
        if model_save_path is not None:
            model_save_path = Path(model_save_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            _joblib.dump(model, model_save_path)  # type: ignore[union-attr]
        
        return TrackingFailureRecommender(model=model)

    @staticmethod
    def load_model(model_path: Path | str) -> DecisionTreeClassifier:
        """
        Load a trained model from disk.

        Parameters
        ----------
        model_path : Path | str
            Path to saved model file (joblib format).

        Returns
        -------
        DecisionTreeClassifier
            Loaded sklearn model.

        Raises
        ------
        ImportError
            If scikit-learn/joblib is not installed.
        FileNotFoundError
            If model file does not exist.

        """
        if not SKLEARN_AVAILABLE:
            err_msg = "scikit-learn and joblib are required to load models. "
            err_msg += "Install with: pip install scikit-learn joblib"
            raise ImportError(err_msg)
        
        model_path = Path(model_path)
        if not model_path.exists():
            err_msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(err_msg)
        
        return _joblib.load(model_path)  # type: ignore[union-attr]

    def save_model(self: Self, model_path: Path | str) -> None:
        """
        Save the current model to disk.

        Parameters
        ----------
        model_path : Path | str
            Path where model will be saved (joblib format).

        Raises
        ------
        ImportError
            If scikit-learn/joblib is not installed.
        ValueError
            If no model is loaded.

        """
        if not SKLEARN_AVAILABLE:
            err_msg = "scikit-learn and joblib are required to save models. "
            err_msg += "Install with: pip install scikit-learn joblib"
            raise ImportError(err_msg)
        
        if self.model is None:
            err_msg = "No model loaded. Cannot save."
            raise ValueError(err_msg)
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _joblib.dump(self.model, model_path)  # type: ignore[union-attr]

    def recommend(
        self: Self,
        tracked_objects: list[TrackedObject],
    ) -> list[tuple[tuple[int, int, int, int], int, str]]:
        """
        Recommend patches for tracking failure detection.

        Parameters
        ----------
        tracked_objects : list[TrackedObject]
            List of tracked objects to evaluate.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], int, str]]
            List of (bbox, priority_score, priority_label) tuples.
            bbox is (x, y, w, h) with padding applied.
            priority_score is numeric (higher = more important).
            priority_label is "high", "medium", or "low".

        """
        patches = []
        
        for obj in tracked_objects:
            # Extract features
            features = np.array([[
                obj.min_eigenvalue,
                obj.ncc,
                obj.acceleration,
                obj.flow_std,
                obj.confidence,
            ]])
            
            # Predict priority
            if self.model is not None:
                priority_label = str(self.model.predict(features)[0])
            else:
                # Heuristic fallback
                priority_label = self._heuristic_priority(obj)
            
            # Map label to numeric score
            priority_map = {"high": 100, "medium": 50, "low": 10}
            priority_score = priority_map.get(priority_label, 10)
            
            # Add padding to bbox
            x, y, w, h = obj.bbox
            pad_w = int(w * self.padding_ratio / 2)
            pad_h = int(h * self.padding_ratio / 2)
            
            padded_bbox = (
                max(0, x - pad_w),
                max(0, y - pad_h),
                w + 2 * pad_w,
                h + 2 * pad_h,
            )
            
            patches.append((padded_bbox, priority_score, priority_label))
        
        return patches

    @staticmethod
    def _heuristic_priority(obj: TrackedObject) -> str:
        """
        Simple heuristic-based priority estimation.

        Parameters
        ----------
        obj : TrackedObject
            The tracked object to evaluate.

        Returns
        -------
        str
            Priority label: "high", "medium", or "low".

        """
        # High priority conditions
        if obj.ncc < 0.5 or obj.acceleration > 10.0 or obj.flow_std > 5.0:
            return "high"
        
        # Medium priority conditions
        if obj.ncc < 0.7 or obj.acceleration > 5.0 or obj.flow_std > 2.0:
            return "medium"
        
        # Low priority (tracking likely successful)
        return "low"

