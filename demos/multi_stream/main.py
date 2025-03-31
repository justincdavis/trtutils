# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Demo showcasing multi-stream YOLO inference using GPU and DLA engines with parallel processing."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2ext
import numpy as np
import time

import trtutils
from trtutils.impls.yolo import YOLO


_log = logging.getLogger(__name__)

# Define video files to process
VIDEO_FILES = [
    Path("videos/mot17_02.mp4"),
    Path("videos/mot17_04.mp4"),
    Path("videos/mot17_05.mp4"),
    Path("videos/mot17_09.mp4"),
    Path("videos/mot17_10.mp4"),
    Path("videos/mot17_11.mp4"),
    Path("videos/mot17_13.mp4"),
]

# Define engine files for different devices
ENGINE_FILES = {
    "gpu": Path("engines/yoloxm_gpu.engine"),
    "dla0": Path("engines/yoloxm_dla0.engine"),
    "dla1": Path("engines/yoloxm_dla1.engine"),
}


@dataclass
class FrameData:
    """Data structure for frame and its metadata."""
    stream_id: int
    frame: np.ndarray
    timestamp: float


@dataclass
class DetectionData:
    """Data structure for detection results."""
    stream_id: int
    bboxes: list[tuple[int, int, int, int]]
    scores: list[float]
    classes: list[int]
    timestamp: float


class FrameQueue:
    """Central queue for managing video frames from all streams."""
    def __init__(self, max_size: int = 30):
        self._queue = deque(maxlen=max_size)
        self._active = 0

    @property
    def active(self) -> bool:
        """Check if the queue is active."""
        return self._active > 0
    
    def increment_stream(self) -> None:
        """Increment the active stream count."""
        self._active += 1

    def decrement_stream(self) -> None:
        """Decrement the active stream count."""
        self._active -= 1

    def push_frame(self, frame_data: FrameData) -> None:
        """Push a frame to the central queue."""
        self._queue.append(frame_data)

    def get_frame(self) -> FrameData | None:
        """Get a frame from the central queue."""
        if self._queue:
            return self._queue.popleft()
        return None


class DetectionQueue:
    """Queue for managing detection results."""
    def __init__(self, max_size: int = 30):
        self.queue = deque(maxlen=max_size)
        self.active = True

    def push_detection(self, detection: DetectionData) -> None:
        """Push detection results to the queue."""
        self.queue.append(detection)

    def get_detection(self) -> DetectionData | None:
        """Get detection results from the queue."""
        if self.queue:
            return self.queue.popleft()
        return None

    def stop(self) -> None:
        """Stop the queue system."""
        self.active = False


class DisplayManager:
    """Manages the display of all video streams and their detections."""
    def __init__(self, num_streams: int):
        self.num_streams = num_streams
        self.frames: dict[int, np.ndarray] = {}
        self.detections: dict[int, tuple[list[tuple[int, int, int, int]], list[float], list[int]]] = {}
        self.active = True

    def update_frame(self, stream_id: int, frame: np.ndarray) -> None:
        """Update a frame for a specific stream."""
        self.frames[stream_id] = frame

    def update_detections(
        self,
        stream_id: int,
        bboxes: list[tuple[int, int, int, int]],
        scores: list[float],
        classes: list[int],
    ) -> None:
        """Update detections for a specific stream."""
        self.detections[stream_id] = (bboxes, scores, classes)

    def get_canvas(self) -> np.ndarray:
        """Get the current canvas with all frames and detections."""
        if not self.frames:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # Calculate grid dimensions
        n_cols = min(3, self.num_streams)
        n_rows = (self.num_streams + n_cols - 1) // n_cols

        # Get frame dimensions
        frame_h, frame_w = next(iter(self.frames.values())).shape[:2]
        canvas = np.zeros((frame_h * n_rows, frame_w * n_cols, 3), dtype=np.uint8)

        # Place frames and detections
        for i, (stream_id, frame) in enumerate(self.frames.items()):
            row = i // n_cols
            col = i % n_cols
            y1, y2 = row * frame_h, (row + 1) * frame_h
            x1, x2 = col * frame_w, (col + 1) * frame_w

            # Copy frame
            canvas[y1:y2, x1:x2] = frame

            # Draw detections if available
            if stream_id in self.detections:
                bboxes, scores, classes = self.detections[stream_id]
                # Adjust bbox coordinates for the grid position
                adjusted_bboxes = [
                    (bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1)
                    for bbox in bboxes
                ]
                cv2ext.bboxes.draw_bboxes(
                    canvas[y1:y2, x1:x2],
                    adjusted_bboxes,
                    scores,
                    classes,
                )

        return canvas

    def stop(self) -> None:
        """Stop the display manager."""
        self.active = False


def create_yolo_models() -> list[YOLO]:
    """Create YOLO models for each device with appropriate preprocessing."""
    models = []
    for device, engine_path in ENGINE_FILES.items():
        try:
            # Use CUDA preprocessing for GPU, CPU for DLAs
            preprocessor = "cuda" if device == "gpu" else "cpu"
            models.append(YOLO(
                engine_path=engine_path,
                preprocessor=preprocessor,
                warmup=True,
            ))
            _log.info(f"Successfully loaded YOLO model for {device} with {preprocessor} preprocessing")
        except Exception as e:
            _log.error(f"Failed to load YOLO model for {device}: {e}")
    return models


def process_stream(
    video_path: Path,
    stream_id: int,
    frame_queue: FrameQueue,
    display_manager: DisplayManager,
) -> None:
    """Process a single video stream and feed frames to the central queue."""
    if not video_path.exists():
        _log.error(f"Video file not found: {video_path}")
        return
    
    frame_queue.increment_stream()

    for i, frame in cv2ext.IterableVideo(video_path):
        if i % 10 == 0:
            print(f"Processing frame {i} of {video_path}")
        frame_data = FrameData(stream_id, frame, time.time())
        frame_queue.push_frame(frame_data)
        display_manager.update_frame(stream_id, frame)
    
    frame_queue.decrement_stream()
    print(f"Finished processing {video_path}")


def process_detections(
    model: YOLO,
    frame_queue: FrameQueue,
    detection_queue: DetectionQueue,
) -> None:
    """Process frames from the central queue using a YOLO model."""
    processed_frames = 0
    while frame_queue.active:
        frame_data = frame_queue.get_frame()
        if frame_data is None:
            continue

        processed_frames += 1

        # Run inference
        dets = model.end2end(frame_data.frame)
        bboxes = [d[0] for d in dets]
        scores = [d[1] for d in dets]
        classes = [d[2] for d in dets]

        # Push detection results
        detection = DetectionData(
            stream_id=frame_data.stream_id,
            bboxes=bboxes,
            scores=scores,
            classes=classes,
            timestamp=frame_data.timestamp,
        )
        detection_queue.push_detection(detection)

    print(f"{model.engine.name} finished processing {processed_frames} frames")


def update_display(
    detection_queue: DetectionQueue,
    display_manager: DisplayManager,
) -> None:
    """Update the display with detection results."""
    with cv2ext.Display("Multi-Stream YOLO Detection") as display:
        while detection_queue.active:
            detection = detection_queue.get_detection()
            if detection is None:
                continue

            display_manager.update_detections(
                detection.stream_id,
                detection.bboxes,
                detection.scores,
                detection.classes,
            )

            canvas = display_manager.get_canvas()
            display.update(canvas)

            if display.stopped:
                detection_queue.stop()
                break


def main() -> None:
    """Run the multi-stream demo with parallel processing."""
    # Set up logging
    trtutils.set_log_level("INFO")

    # Create YOLO models for each device
    models = create_yolo_models()

    # Create queues and display manager
    frame_queue = FrameQueue(max_size=1000)
    detection_queue = DetectionQueue(max_size=1000)
    display_manager = DisplayManager(len(VIDEO_FILES))

    # Start threads
    video_threads = []
    
    # Start frame feeding threads
    for i, video_path in enumerate(VIDEO_FILES):
        thread = threading.Thread(
            target=process_stream,
            args=(video_path, i, frame_queue, display_manager),
        )
        thread.daemon = True
        thread.start()
        video_threads.append(thread)

    # Start detection processing threads for each model
    detection_threads = []
    for model in models:
        thread = threading.Thread(
            target=process_detections,
            args=(model, frame_queue, detection_queue),
        )
        thread.daemon = True
        thread.start()
        detection_threads.append(thread)

    # # Start display update thread
    # display_thread = threading.Thread(
    #     target=update_display,
    #     args=(detection_queue, display_manager),
    # )
    # display_thread.daemon = True
    # display_thread.start()
    # threads.append(display_thread)

    # Wait for all threads to complete
    for thread in video_threads:
        print(f"Waiting for video thread {thread.name} to finish")
        thread.join()
    for thread in detection_threads:
        print(f"Waiting for detection thread {thread.name} to finish")
        thread.join()


if __name__ == "__main__":
    main()
