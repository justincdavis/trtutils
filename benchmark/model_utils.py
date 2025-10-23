# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
# MIT License
"""Utility functions for managing benchmark models."""

from __future__ import annotations

from pathlib import Path

REPO_DIR = Path(__file__).parent.parent


def ensure_model_available(
    model_name: str,
    imgsz: int,
    model_to_dir: dict[str, str],
    opset: int = 17,
    *,
    auto_download: bool = True,
) -> Path:
    """Ensure a model is available, downloading it if necessary."""
    from trtutils.download import download

    if model_name not in model_to_dir:
        err_msg = f"Unknown model: {model_name}. Not found in model directory mapping."
        raise ValueError(err_msg)
    
    model_dir = REPO_DIR / "data" / model_to_dir[model_name]
    model_path = model_dir / f"{model_name}_{imgsz}.onnx"
    
    if model_path.exists():
        return model_path
    
    if not auto_download:
        err_msg = f"Model not found: {model_path}"
        raise FileNotFoundError(err_msg)
    
    # Auto-download
    print(f"Downloading {model_name} @ {imgsz}x{imgsz}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    download(
        model=model_name,
        output=model_path,
        opset=opset,
        imgsz=imgsz,
        accept=True,
        verbose=False,
    )
    return model_path


def build_model(
    onnx: Path,
    output: Path,
    imgsz: int,
    opt_level: int = 3,
) -> None:
    """Build a model from an ONNX file."""
    from trtutils.builder import build_engine
    from trtutils.builder.hooks import yolo_efficient_nms_hook

    # define the shapes, all yolo models have the "images" shape
    shapes = []
    if "rfdetr" in onnx.stem:
        shapes.append(
            ("input", (1, 3, imgsz, imgsz)),
        )
    elif "rtdetrv1" in onnx.stem or "rtdetrv2" in onnx.stem or "deim" in onnx.stem:
        shapes.extend([
            ("image", (1, 3, imgsz, imgsz)),
            ("orig_image_size", (1, 2)),
        ])
    elif "rtdetrv3" in onnx.stem:
        shapes.extend([
            ("image", (1, 3, imgsz, imgsz)),
            ("im_shape", (1, 2)),
            ("scale_factor", (1, 2)),
        ])
    else:
        shapes.append(
            ("images", (1, 3, imgsz, imgsz)),
        )

    # setup the hooks
    # setup the shapes based on the modelname
    yolo_add_nms = ["yolov8", "yolov11", "yolov12", "yolov13", "yolox"]

    # only yolo models may need NMS hook
    hooks = []
    if sum(1 for m in yolo_add_nms if m in onnx.stem) > 0:
        hooks.append(yolo_efficient_nms_hook())

    build_engine(
        onnx=onnx,
        output=output,
        fp16=True,
        optimization_level=opt_level,
        shapes=shapes,
        hooks=hooks,
    )
