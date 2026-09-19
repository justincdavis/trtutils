# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""File showcasing the HOIDETR hand-object interaction detector."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import set_log_level
from trtutils.models import HOIDETR

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    onnx_path = DATA_DIR / "hoi_detr" / "hoi_detr_vitl_640.onnx"
    engine_path = DATA_DIR / "hoi_detr" / "hoi_detr_vitl_640.engine"

    if not onnx_path.exists():
        print("Downloading HOI-DETR ONNX model...")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        HOIDETR.download("hoi_detr_vitl", onnx_path)

    if not engine_path.exists():
        HOIDETR.build(onnx_path, engine_path)

    image_path = DATA_DIR / "people.jpeg"
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise FileNotFoundError(msg)

    model = HOIDETR(engine_path, warmup=True)

    t0 = time.perf_counter()
    interactions = model.end2end(image)
    t1 = time.perf_counter()
    print(f"Inference time: {round((t1 - t0) * 1000.0, 2)} ms")
    print(f"Found {len(interactions)} hand(s)")

    for hand, obj, second, _side, _contact in interactions:
        hand_bbox, _hand_score = hand
        hx1, hy1, hx2, hy2 = hand_bbox
        cv2.rectangle(image, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        hand_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)

        if obj is not None:
            obj_bbox, _obj_score = obj
            ox1, oy1, ox2, oy2 = obj_bbox
            cv2.rectangle(image, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
            obj_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
            cv2.line(image, hand_center, obj_center, (0, 255, 0), 2)

        if second is not None:
            second_bbox, _second_score = second
            sx1, sy1, sx2, sy2 = second_bbox
            cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)

    output_path = DATA_DIR / "people_hands.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
