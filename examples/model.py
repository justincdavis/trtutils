# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing an example usecase of the TRTModel class."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trtutils import TRTModel


# This example shows how to use the TRTModel class
# on running a Yolo model with a single input image.
# The Yolo model is not included in this repository.
# This works with a yolov7 engine created by
# using the export script locating in the yolov7 repository.
# Then generate an engine using TensorRT by:
#  trtexec --onnx=yolo.onnx --saveEngine=yolo.engine
# The resulting engine can be used with this example.
def main() -> None:
    def preprocess(inputs: list[np.ndarray]) -> list[np.ndarray]:
        def _process(img: np.ndarray) -> np.ndarray:
            # all calls to binary so it is fast
            img = cv2.resize(img, (640, 640), cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255  # type: ignore[assignment]
            img = img[np.newaxis, :]
            img = np.transpose(img, (0, 3, 1, 2))
            return np.ascontiguousarray(img, dtype=np.float32)  # type: ignore[no-any-return]

        return [_process(img) for img in inputs]

    def postprocess(outputs: list[np.ndarray]) -> list[np.ndarray]:
        # get the highest scoring bbox
        results = []
        for output in outputs:
            score = output[2][0][0]
            bbox = output[1][0][0]
            cls = output[0][0][0]
            results.append(np.array((score, bbox, cls)))
        return results

    model = TRTModel(
        "yolo.engine",
        preprocess,
        postprocess,
        warmup=True,
    )

    img_path = str(Path(__file__).parent.parent / "data" / "horse.jpg")
    img = cv2.imread(img_path)
    if img is None:
        err_msg = f"Failed to load image from {img_path}"
        raise FileNotFoundError(err_msg)

    outputs = model.run([img])
    # OR
    # outputs = model([img])
    # OR
    # imgs = model.preprocess([img])
    # outputs = model.run(imgs, preprocessed=True)

    print(outputs)  # [score, bbox, cls]


if __name__ == "__main__":
    main()
