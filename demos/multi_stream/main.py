# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Demo showcasing multi-stream YOLO inference using GPU and DLA engines with parallel processing."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from queue import Queue, Full, Empty

import cv2
import cv2ext
import trtutils
import numpy as np


VIDEO_FILES = [
    Path("videos/mot17_02.mp4"),
    Path("videos/mot17_04.mp4"),
    Path("videos/mot17_05.mp4"),
    Path("videos/mot17_09.mp4"),
    Path("videos/mot17_10.mp4"),
    Path("videos/mot17_11.mp4"),
    Path("videos/mot17_13.mp4"),
]

ENGINE_FILES = {
    "gpu": Path("engines/yoloxm_gpu.engine"),
    "dla0": Path("engines/yoloxm_dla0.engine"),
    "dla1": Path("engines/yoloxm_dla1.engine"),
}

FINISHED = 0


def _feed_frames(
    video: cv2ext.IterableVideo,
    stream_id: int,
    in_queue: Queue[tuple[int, int, np.ndarray]],
) -> None:
    for i, frame in video:
        while True:
            try:
                in_queue.put((stream_id, i, frame), timeout=0.25)
                # print(f"Feeding frame {i} from stream {stream_id}")
                break
            except Full:
                # print(f"Stream {stream_id} queue is full, waiting for 0.01s")
                time.sleep(0.01)
    global FINISHED
    FINISHED += 1


def _process_frames(
    yolo: trtutils.impls.yolo.YOLO,
    in_queue: Queue[tuple[int, int, np.ndarray]],
    out_queue: Queue[tuple[int, int, np.ndarray, list[tuple[tuple[int, int, int, int], float, int]]]],
) -> None:
    while FINISHED < len(VIDEO_FILES):
        try:
            stream_id, frame_id, frame = in_queue.get(timeout=0.25)
            # print(f"Processing frame {frame_id} from stream {stream_id}")
            dets = yolo.end2end(frame)
            out_queue.put((stream_id, frame_id, frame, dets))
        except Empty:
            # print(f"Stream {stream_id} queue is empty, waiting for 0.25s")
            time.sleep(0.25)
    # print(f"Stream {stream_id} finished")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", action="store_true", help="Enable display output")
    parser.add_argument("--dla0", action="store_true", help="Use DLA0 engine")
    parser.add_argument("--dla1", action="store_true", help="Use DLA1 engine")
    args = parser.parse_args()

    engines_files = [ENGINE_FILES["gpu"]]
    if args.dla0:
        engines_files.append(ENGINE_FILES["dla0"])
    if args.dla1:
        engines_files.append(ENGINE_FILES["dla1"])

    yolos = [
        trtutils.impls.yolo.YOLO(engine_file, input_range=(0, 255), conf_thres=0.1, warmup_iterations=10, warmup=True)
        for engine_file in engines_files
    ]
    videos = [
        cv2ext.IterableVideo(video_file)
        for video_file in VIDEO_FILES
    ]
    total_frames = sum(len(video) for video in videos)
    
    in_queue = Queue(maxsize=100)
    out_queue = Queue(maxsize=100 if args.display else 0)

    video_threads = [
        threading.Thread(target=_feed_frames, args=(video, i, in_queue))
        for i, video in enumerate(videos)
    ]
    yolo_threads = [
        threading.Thread(target=_process_frames, args=(yolo, in_queue, out_queue))
        for yolo in yolos
    ]

    if args.display:
        window_names = [
            f"Stream {i}"
            for i in range(len(videos))
        ]
    else:
        window_names = None

    t00 = time.time()
    for thread in video_threads + yolo_threads:
        thread.start()

    if window_names:
        counter = 0
        while FINISHED < len(VIDEO_FILES):
            stream_id, frame_id, frame, dets = out_queue.get(timeout=0.25)
            # print(f"Frame {frame_id} from stream {stream_id} fetched")
            canvas = cv2ext.detection.draw_detections(frame, dets)
            canvas = cv2ext.image.draw.text(canvas, str(frame_id), (10, 30), color=(0, 255, 0))
            cv2.imshow(window_names[stream_id], canvas)
            if counter % 10 == 0:
                cv2.waitKey(1)
            counter += 1

    for thread in video_threads + yolo_threads:
        thread.join()
    t11 = time.time()

    print(f"Processing {total_frames} frames took {t11 - t00:.2f} seconds")
    print(f"Total FPS: {total_frames / (t11 - t00):.2f}")

    if window_names:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _main()
