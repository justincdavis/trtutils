# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.download._tools import (
    get_weights_cache_dir,
    git_clone,
    handle_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path


def export_depth_anything_v2(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "DepthAnythingV2 is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    imgsz = handle_imgsz(imgsz, 518, "DepthAnythingV2", enforce=True)
    git_clone(
        "https://github.com/DepthAnything/Depth-Anything-V2",
        directory,
        "e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1",
        no_cache=no_cache,
        verbose=verbose,
    )
    da_v2_dir = directory / "Depth-Anything-V2"
    run_uv_pip_install(
        da_v2_dir,
        bin_path.parent,
        "depth_anything_v2",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(da_v2_dir, config, python_path, no_cache=no_cache, verbose=verbose)

    # Handle caching of depth_anything_v2 downloaded .pt weights
    weights_cache_dir = get_weights_cache_dir()
    cached_pth_file = weights_cache_dir / config["weights"]
    target_pth_file = directory / config["weights"]

    # Check if .pth file exists in cache
    if cached_pth_file.exists() and not no_cache:
        LOG.info(f"Using cached depth_anything_v2 weights: {config['weights']}")
        shutil.copy(cached_pth_file, target_pth_file)

    # get the encoder from the model name
    encoders = {
        "depth_anything_v2_small": "vits",
        "depth_anything_v2_base": "vitb",
        "depth_anything_v2_large": "vitl",
        "depth_anything_v2_giant": "vitg",
    }
    encoder = encoders[model]

    program = f"""
import torch

from depth_anything_v2.dpt import DepthAnythingV2

encoder = "{encoder}"

model_configs = {{
    'vits': {{'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}},
    'vitb': {{'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}},
    'vitl': {{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}},
    'vitg': {{'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}},
}}

depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load('depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to('cpu').eval()

dummy_input = torch.ones((3, {imgsz}, {imgsz})).unsqueeze(0)

example_output = depth_anything.forward(dummy_input)

onnx_path = f'depth_anything_v2_{encoder}.onnx'

torch.onnx.export(
    depth_anything,
    dummy_input,
    onnx_path,
    opset_version={opset},
    input_names=['input'],
    output_names=['output'],
    verbose=True,
    dynamic_axes={{'input': {{0: 'batch_size'}}, 'output': {{0: 'batch_size'}}}},
)
"""
    run_cmd(
        [
            python_path,
            "-c",
            program,
        ],
        cwd=da_v2_dir,
        verbose=verbose,
    )

    # Cache the .pth file if it was downloaded and not already cached
    if target_pth_file.exists() and not cached_pth_file.exists():
        shutil.copy(target_pth_file, cached_pth_file)
        LOG.info(f"Cached DepthAnythingV2 weights: {config['weights']}")

    model_path = da_v2_dir / f"depth_anything_v2_{encoder}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
