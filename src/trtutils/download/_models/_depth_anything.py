# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import os
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


_DA_V1_COMMIT = "1d03336771fe09c5398ffdd211441e33941a97dc"
_DA_V3_COMMIT = "41736238f5bced4debf3f2a12375d2466874866d"
_DA_V3_MIN_OPSET = 20


def export_depth_anything_v1(
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
            "DepthAnythingV1 is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    imgsz = handle_imgsz(imgsz, 518, "DepthAnythingV1", enforce=True)
    git_clone(
        "https://github.com/LiheYoung/Depth-Anything",
        directory,
        _DA_V1_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    da_v1_dir = directory / "Depth-Anything"
    run_uv_pip_install(
        da_v1_dir,
        bin_path.parent,
        "depth_anything_v1",
        no_cache=no_uv_cache,
        verbose=verbose,
    )

    encoder = config["encoder"]

    program = f"""
import torch

from depth_anything.dpt import DepthAnything

encoder = "{encoder}"

depth_anything = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{{encoder}}14")
depth_anything = depth_anything.to("cpu").eval()

dummy_input = torch.ones((3, {imgsz}, {imgsz})).unsqueeze(0)

example_output = depth_anything.forward(dummy_input)

onnx_path = f"depth_anything_v1_{{encoder}}.onnx"

torch.onnx.export(
    depth_anything,
    dummy_input,
    onnx_path,
    opset_version={opset},
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={{"input": {{0: "batch_size"}}, "output": {{0: "batch_size"}}}},
    dynamo=False,
)
"""
    run_cmd(
        [
            python_path,
            "-c",
            program,
        ],
        cwd=da_v1_dir,
        verbose=verbose,
    )

    model_path = da_v1_dir / f"depth_anything_v1_{encoder}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


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


def export_depth_anything_v3(
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
            "DepthAnythingV3 monocular variants are Apache-2.0 licensed, be aware of license "
            "restrictions. ONNX export uses CUDA when available and falls back to CPU; the CPU "
            "path works but is significantly slower for the LARGE checkpoints."
        )
    # ros2-depth-anything-v3-trt pins opset 20 for V3.
    opset = max(opset, _DA_V3_MIN_OPSET)
    imgsz = handle_imgsz(imgsz, 518, "DepthAnythingV3", adjust_div=14)
    git_clone(
        "https://github.com/ByteDance-Seed/depth-anything-3",
        directory,
        _DA_V3_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    da_v3_dir = directory / "depth-anything-3"

    # Two patches to V3's api.py:
    #   1. Force fp16 autocast (ONNX does not fully support bfloat16). See
    #      https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt/blob/main/onnx/README.md
    #   2. Drop the eager `from depth_anything_3.utils.export import export` import. The
    #      monocular forward path does not need it, but it transitively imports moviepy,
    #      pycolmap, plyfile, etc. — including pycolmap, which has no aarch64 wheels.
    api_path = da_v3_dir / "src" / "depth_anything_3" / "api.py"
    api_src = api_path.read_text(encoding="utf-8")
    bf16_line = (
        "autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16"
    )
    fp16_line = "autocast_dtype = torch.float16"
    export_import_line = "from depth_anything_3.utils.export import export"
    for required in (bf16_line, export_import_line):
        if required not in api_src:
            err_msg = (
                f"Failed to apply DepthAnythingV3 ONNX patch: expected line not found "
                f"in {api_path}: {required!r}. The pinned commit ({_DA_V3_COMMIT}) may "
                f"need to be updated."
            )
            raise RuntimeError(err_msg)
    api_src = api_src.replace(bf16_line, fp16_line)
    api_src = api_src.replace(export_import_line, f"# patched out: {export_import_line}")
    api_path.write_text(api_src, encoding="utf-8")

    run_uv_pip_install(
        da_v3_dir,
        bin_path.parent,
        "depth_anything_v3",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    # Use PYTHONPATH to expose the cloned package without `pip install -e .`. The V3
    # pyproject pulls heavy deps (pycolmap, open3d, xformers, etc.) which are not all
    # needed for ONNX export and lack wheels on some platforms (e.g. aarch64).
    pythonpath = str((da_v3_dir / "src").resolve())

    # Cache the HF snapshot under our weights cache so re-runs are fast.
    weights_cache_dir = get_weights_cache_dir()
    snapshot_cache = weights_cache_dir / model
    repo_id = config["repo_id"]
    snapshot_dir = directory / model

    if snapshot_cache.exists() and not no_cache:
        LOG.info(f"Using cached DepthAnythingV3 snapshot: {model}")
        shutil.copytree(snapshot_cache, snapshot_dir)
    else:
        LOG.info(f"Downloading DepthAnythingV3 snapshot from HF Hub: {repo_id}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                python_path,
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    f'snapshot_download(repo_id="{repo_id}", local_dir="{snapshot_dir}")'
                ),
            ],
            cwd=da_v3_dir,
            verbose=verbose,
        )
        if snapshot_cache.exists():
            shutil.rmtree(snapshot_cache)
        shutil.copytree(snapshot_dir, snapshot_cache)
        LOG.info(f"Cached DepthAnythingV3 snapshot: {model}")

    onnx_filename = f"{model}.onnx"
    program = f"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from torch import nn

from depth_anything_3.api import DepthAnything3


class DepthAnything3OnnxWrapper(nn.Module):
    def __init__(self, api_model):
        super().__init__()
        self.model = api_model

    def forward(self, image):
        out = self.model(
            image.unsqueeze(1),
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
        )
        return out["depth"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api_model = DepthAnything3.from_pretrained(r"{snapshot_dir}")
api_model = api_model.to(device).eval()

wrapper = DepthAnything3OnnxWrapper(api_model).to(device)
dummy_input = torch.zeros(1, 3, {imgsz}, {imgsz}, device=device)

with torch.no_grad():
    _ = wrapper(dummy_input)
    torch.onnx.export(
        wrapper,
        dummy_input,
        r"{onnx_filename}",
        export_params=True,
        opset_version={opset},
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={{"input": {{0: "batch_size"}}, "output": {{0: "batch_size"}}}},
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
"""
    env = {**os.environ, "PYTHONPATH": pythonpath}
    run_cmd(
        [python_path, "-c", program],
        cwd=da_v3_dir,
        env=env,
        verbose=verbose,
    )

    model_path = da_v3_dir / onnx_filename
    new_model_path = directory / onnx_filename
    shutil.move(model_path, new_model_path)
    return new_model_path
