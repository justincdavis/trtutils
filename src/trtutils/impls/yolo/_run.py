# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2ext
import numpy as np

from trtutils._log import set_log_level

from ._yolo import YOLO


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        "Evaluate a model on an input file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="The path to the TensorRT engine file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="The path to the input file.",
    )
    parser.add_argument(
        "--input_high",
        type=int,
        default=1,
        help="The high value which input should be.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="The confidence threshold to filter detections.",
    )
    parser.add_argument(
        "--buffersize",
        type=int,
        default=10,
        help="The amount of frames to buffer performance times for dispaly.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Output debug information from the execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output additional debugging information.",
    )
    args = parser.parse_args()

    if args.debug:
        set_log_level("DEBUG")

    input_range = (0.0, float(args.input_high))
    yolo = YOLO(
        args.model,
        input_range=input_range,
        warmup_iterations=10,
        warmup=True,
        preprocessor="cuda",
        resize_method="letterbox",
        conf_thres=args.conf_thresh,
    )

    buffers: dict[str, deque[float]] = {
        "PRE": deque(maxlen=args.buffersize),
        "INF": deque(maxlen=args.buffersize),
        "POST": deque(maxlen=args.buffersize),
        "DEC": deque(maxlen=args.buffersize),
        "TOTAL": deque(maxlen=args.buffersize),
    }

    # for each frame in the scene
    image: np.ndarray
    with cv2ext.Display(
        f"YOLO: {yolo.name}",
        stopkey="q",
        buffersize=args.buffersize,
    ) as display:
        for _, image in cv2ext.IterableVideo(args.input):
            if display.stopped:
                break

            t0 = time.time()
            tensor, ratio, padding = yolo.preprocess(
                image,
                no_copy=True,
                verbose=args.verbose,
            )
            t1 = time.time()
            outputs = yolo.run(
                tensor,
                preprocessed=True,
                postprocess=False,
                no_copy=True,
                verbose=args.verbose,
            )
            t2 = time.time()
            postoutputs = yolo.postprocess(
                outputs,
                ratio,
                padding,
                no_copy=True,
                verbose=args.verbose,
            )
            t3 = time.time()
            detections = yolo.get_detections(postoutputs, verbose=args.verbose)
            bboxes = [bbox for (bbox, _, _) in detections]
            t4 = time.time()

            pretime = t1 - t0
            inftime = t2 - t1
            posttime = t3 - t2
            detecttime = t4 - t3

            # convert times to milliseconds
            pretime *= 1000.0
            inftime *= 1000.0
            posttime *= 1000.0
            detecttime *= 1000.0

            # add the times to the buffers
            buffers["PRE"].append(pretime)
            buffers["INF"].append(inftime)
            buffers["POST"].append(posttime)
            buffers["DEC"].append(detecttime)
            buffers["TOTAL"].append(pretime + inftime + posttime + detecttime)

            canvas = cv2ext.bboxes.draw_bboxes(image, bboxes)
            canvas = cv2ext.image.draw.text(
                canvas,
                f"PRE: {round(float(np.mean(list(buffers['PRE']))), 1)} ms",
                (10, 40),
            )
            canvas = cv2ext.image.draw.text(
                canvas,
                f"INF: {round(float(np.mean(list(buffers['INF']))), 1)} ms",
                (10, 70),
            )
            canvas = cv2ext.image.draw.text(
                canvas,
                f"POST: {round(float(np.mean(list(buffers['POST']))), 1)} ms",
                (10, 100),
            )
            canvas = cv2ext.image.draw.text(
                canvas,
                f"DEC: {round(float(np.mean(list(buffers['DEC']))), 1)} ms",
                (10, 130),
            )
            canvas = cv2ext.image.draw.text(
                canvas,
                f"TOTAL: {round(float(np.mean(list(buffers['TOTAL']))), 1)} ms",
                (10, 180),
            )
            display.update(canvas)

    del yolo
