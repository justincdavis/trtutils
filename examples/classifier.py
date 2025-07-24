# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the Classifier class."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.image import Classifier


def main() -> None:
    engine_dir = Path(__file__).parent.parent / "data" / "engines"
    engines = [
        engine_dir / "resnet18.engine",
    ]

    img_path = str(Path(__file__).parent.parent / "data" / "horse.jpg")
    img = cv2.imread(img_path)
    if img is None:
        err_msg = f"Failed to load image from {img_path}"
        raise FileNotFoundError(err_msg)

    for engine in engines:
        classifier = Classifier(engine, warmup=True, preprocessor="cuda")
        print(classifier.name)

        t0 = time.perf_counter()
        output = classifier.run(img)
        classifications = classifier.get_classifications(output, top_k=5)
        t1 = time.perf_counter()

        print(
            f"RUN, classifications: {classifications}, in {round((t1 - t0) * 1000.0, 2)}"
        )

        # OR

        # end2end makes a few memory optimzations by avoiding extra GPU
        # memory transfers
        t0 = time.perf_counter()
        classifications = classifier.end2end(img, top_k=5)
        t1 = time.perf_counter()

        print(
            f"END2END: classifications: {classifications}, in {round((t1 - t0) * 1000.0, 2)}"
        )

        del classifier


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
