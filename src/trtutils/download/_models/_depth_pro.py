# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from trtutils.download._tools import (
    git_clone,
    handle_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path

_DP_COMMIT = "9e65e4dbe9568d23c546fcec53302b10445e109e"
# a 1536 image as a pyramid of 384-pixel patches: the size is fixed
_DP_IMGSZ = 1536
_DP2_COMMIT = "aa389cbb0b74341664ed663a0966c859905cd6c0"
# the encoder and decoder work on multiples of 32 (patch 16, a stride 2 level below it)
_DP2_DIVISOR = 32
_DP2_DEFAULT_HW = (576, 1024)

# export script code shared by the Depth Pro exporters
_SAVE_ONNX = """
def save_onnx(module, dummy, output_path, opset):
    # over 2 GB, torch writes one file per weight: export to a scratch directory, then
    # free it before the weights are written again, to one sidecar next to the onnx
    os.makedirs("raw", exist_ok=True)
    raw_path = os.path.join("raw", output_path)
    torch.onnx.export(
        module,
        dummy,
        raw_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["depth", "focal_length"],
        dynamo=False,
    )
    onnx_model = onnx.load(raw_path)
    shutil.rmtree("raw")
    weight_bytes = sum(tensor.ByteSize() for tensor in onnx_model.graph.initializer)
    if weight_bytes < onnx.checker.MAXIMUM_PROTOBUF:
        onnx.save(onnx_model, output_path)
        return
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path + ".data",
    )
"""


def _hw_multiple(imgsz: int | tuple[int, int] | None, default: tuple[int, int]) -> tuple[int, int]:
    if imgsz is None:
        return default
    height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    if height % _DP2_DIVISOR or width % _DP2_DIVISOR:
        err_msg = f"Depth Pro 2 needs imgsz divisible by {_DP2_DIVISOR}, got {imgsz}"
        raise ValueError(err_msg)
    return height, width


def export_depth_pro_2(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | tuple[int, int] | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    height, width = _hw_multiple(imgsz, _DP2_DEFAULT_HW)
    git_clone(
        "https://github.com/apple-aiml-research/ml-depth-pro-2",
        directory,
        _DP2_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    repo_dir = directory / "ml-depth-pro-2"
    run_uv_pip_install(
        repo_dir,
        bin_path.parent,
        "depth_pro_2",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    # upstream loads ./checkpoints/<name> before its own download, so fetch both there
    checkpoints = repo_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for url in (config["url"], config["fov_url"]):
        run_download(checkpoints, {"url": url}, python_path, no_cache=no_cache, verbose=verbose)

    output_name = f"{model}.onnx"
    script_content = f"""\
import math
import os
import shutil

import numpy as np
import onnx
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

import depth_pro_2.network.fov as fov_module
from depth_pro_2 import create_model_and_transforms
from depth_pro_2.network.fov import MAX_ASPECT

height, width = {height}, {width}
output_path = "{output_name}"
# exports never need gradients
torch.set_grad_enabled(False)

RAD = math.pi / 180.0
FP16_PEAK = 2.0**14
EXP_UNDERFLOW_LOGIT = -128.0


def resize_matrix(n_in, n_out):
    # upstream's antialiased bilinear resize along one axis, as a (n_out, n_in) matrix
    eye = torch.eye(n_in).view(n_in, 1, 1, n_in)
    return F.interpolate(eye, size=(1, n_out), mode="bilinear", antialias=True)[:, 0, 0, :].T


class DepthPro2Image(nn.Module):
    # one image (no temporal state) to metric depth and the estimated focal length in pixels;
    # DepthPro2.infer and FOVEstimator.forward / focal_length_px at a static size
    def __init__(self, model):
        super().__init__()
        self.model = model
        fov = model.fov
        new_h, new_w = fov.input_size(height, width)
        self.register_buffer("resize_h", resize_matrix(height, new_h))
        self.register_buffer("resize_w", resize_matrix(width, new_w).T.contiguous())
        self.full = (new_h, new_w)
        self.crop = None
        lo, hi = min(new_h, new_w), max(new_h, new_w)
        if hi > MAX_ASPECT * lo:
            keep = max(32, int(MAX_ASPECT * lo) // 32 * 32)
            self.crop = ((hi - keep) // 2, keep, new_w > new_h)

    def forward(self, x):
        model, fov = self.model, self.model.fov
        inverse_depth = model.forward(x)
        y = self.resize_h @ x @ self.resize_w
        if self.crop is not None:
            off, keep, wide = self.crop
            y = y[..., off : off + keep] if wide else y[..., off : off + keep, :]
        tokens, grid = fov.encode(y)
        logits = fov.head(tokens, grid)
        fov_deg = fov.min_fov_deg + torch.sigmoid(logits) * (fov.max_fov_deg - fov.min_fov_deg)
        if self.crop is not None:
            # crop_angles_to_full without deg2rad / rad2deg, which have no onnx export
            crop_h, crop_w = y.shape[-2:]
            full_h, full_w = self.full
            src_h, src_w = crop_h * height / full_h, crop_w * width / full_w
            fx = (src_w / 2) / torch.tan(fov_deg[..., 0] * RAD / 2)
            fy = (src_h / 2) / torch.tan(fov_deg[..., 1] * RAD / 2)
            fov_deg = torch.stack(
                [2 * torch.atan((width / 2) / fx), 2 * torch.atan((height / 2) / fy)], dim=-1
            ) / RAD
        # the focal length from the field of view of the shorter side
        if width < height:
            focal_length = 0.5 * width / torch.tan(fov_deg[:, 0:1] * (RAD / 2))
        else:
            focal_length = 0.5 * height / torch.tan(fov_deg[:, 1:2] * (RAD / 2))
        depth = focal_length.view(-1, 1, 1, 1) / (width * inverse_depth)
        return depth, focal_length


# the fov head's sin-cos code of the static grid as a float32 constant: traced, its python
# scalar math becomes float64, which onnxruntime and tensorrt reject
_sincos = fov_module.sincos_2d
_sincos_cache = {{}}


def sincos_constant(gh, gw, dim, device):
    key = (int(gh), int(gw), dim)
    if key not in _sincos_cache:
        _sincos_cache[key] = _sincos(*key, device).float()
    return _sincos_cache[key]


fov_module.sincos_2d = sincos_constant

# fp16: the first dinov3 blocks reach attention logits of ~2.6e6, past the fp16 maximum. as
# upstream's core ml export (demo/export/fp16.py), a block whose unscaled logits exceed 2**14
# on the example image computes them divided by a power of two, subtracts the row max (softmax
# is unchanged by it) and multiplies back: the same attention, in fp16 range
_sdpa = F.scaled_dot_product_attention
_state = {{"divisor": None, "peak": None}}


def attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if _state["peak"] is not None:
        _state["peak"] = max(_state["peak"], float((q @ k.transpose(-1, -2)).abs().max()))
    divisor = _state["divisor"]
    if divisor is None:
        return _sdpa(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    logits = (q * (scale / divisor)) @ k.transpose(-1, -2)
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    shifted = shifted.clamp(min=EXP_UNDERFLOW_LOGIT / divisor, max=0.0)
    return (shifted * divisor).softmax(dim=-1) @ v


F.scaled_dot_product_attention = attention

model, transform = create_model_and_transforms("{config["preset"]}", "cpu", torch.float32)
wrapper = DepthPro2Image(model).eval()
example = Image.open("assets/example.jpg").convert("RGB").resize((width, height), Image.BILINEAR)
example = transform(np.asarray(example))[None]
expected = wrapper(example)


def fitting_divisor(peak):
    return 2.0 ** max(0, math.ceil(math.log2(peak / FP16_PEAK)))


# fp16: the residual stream of the large encoders peaks at ~1.3e5, past the fp16 maximum. as
# upstream's divide_residual_stream, divide everything that writes the stream by a power of
# two and each norm eps by its square (the norms are scale invariant), and multiply the convs
# that read the tapped tokens back
encoder = model.encoder
stream_peak = 0.0


def record_stream(module, args):
    global stream_peak
    stream_peak = max(stream_peak, float(args[0].abs().max()))


handles = [
    norm.register_forward_pre_hook(record_stream)
    for block in encoder.blocks
    for norm in (block.norm1, block.norm2)
]
wrapper(example)
for handle in handles:
    handle.remove()
stream_divisor = fitting_divisor(stream_peak)
print(f"residual stream peak {{stream_peak:.0f}}, divisor {{stream_divisor:g}}")
for token in (encoder.cls_token, encoder.reg_token):
    token /= stream_divisor
writers = [encoder.patch_embed.proj]
writers += [m for block in encoder.blocks for m in (block.attn.proj, block.mlp.fc2)]
for writer in writers:
    writer.weight /= stream_divisor
    writer.bias /= stream_divisor
for block in encoder.blocks:
    block.norm1.eps /= stream_divisor**2
    block.norm2.eps /= stream_divisor**2
for layer in encoder.reassemble:
    layer[0].weight *= stream_divisor

peaks, divisors = {{}}, {{}}


def set_divisor(name):
    def hook(module, args):
        _state["divisor"] = divisors.get(name)
        _state["peak"] = 0.0 if name not in divisors else None

    return hook


def record_peak(name):
    def hook(module, args, output):
        if _state["peak"] is not None:
            peaks[name] = max(peaks.get(name, 0.0), _state["peak"])
        _state["divisor"], _state["peak"] = None, None

    return hook


for name, module in model.named_modules():
    if name.endswith(".attn"):
        module.register_forward_pre_hook(set_divisor(name))
        module.register_forward_hook(record_peak(name))

wrapper(example)
divisors = {{name: fitting_divisor(peak) for name, peak in peaks.items() if peak > FP16_PEAK}}
print(f"attention logit divisors: {{divisors}}")
# fails loudly if the rewrites change the outputs; compared inverted, as the depth of the sky
# is huge or infinite
for before, after in zip(expected, wrapper(example)):
    torch.testing.assert_close(1 / after, 1 / before, rtol=1e-3, atol=1e-5)

{_SAVE_ONNX}
save_onnx(wrapper, torch.zeros(1, 3, height, width), output_path, {opset})
"""
    script_path = repo_dir / "_export_depth_pro_2.py"
    script_path.write_text(script_content)
    env = {**os.environ, "PYTHONPATH": str((repo_dir / "src").resolve())}
    run_cmd([python_path, str(script_path)], cwd=repo_dir, env=env, verbose=verbose)
    return repo_dir / output_name


def export_depth_pro(
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
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # the patch pyramid (384-pixel patches of a 1536 image) fixes the input size
    handle_imgsz(imgsz, _DP_IMGSZ, "DepthPro", enforce=True)
    git_clone(
        "https://github.com/apple/ml-depth-pro",
        directory,
        _DP_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    repo_dir = directory / "ml-depth-pro"
    run_uv_pip_install(
        repo_dir,
        bin_path.parent,
        "depth_pro",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    # upstream loads ./checkpoints/depth_pro.pt
    checkpoints = repo_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    run_download(checkpoints, config, python_path, no_cache=no_cache, verbose=verbose)

    output_name = f"{model}.onnx"
    script_content = f"""\
import math
import os
import shutil

import onnx
import torch
from torch import nn

from depth_pro import create_model_and_transforms

output_path = "{output_name}"
# exports never need gradients
torch.set_grad_enabled(False)
RAD = math.pi / 180.0


class DepthProImage(nn.Module):
    # DepthPro.infer at the network size: metric depth and the focal length in pixels
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.size = model.img_size

    def forward(self, x):
        canonical_inverse_depth, fov_deg = self.model(x)
        # deg2rad has no onnx export
        focal_length = 0.5 * self.size / torch.tan(0.5 * fov_deg.view(-1, 1) * RAD)
        inverse_depth = canonical_inverse_depth * (self.size / focal_length.view(-1, 1, 1, 1))
        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
        return depth, focal_length


model, _ = create_model_and_transforms()
wrapper = DepthProImage(model.eval()).eval()
{_SAVE_ONNX}
# three vit-l encoders: over 2 GB, so the weights go to a sidecar
save_onnx(wrapper, torch.zeros(1, 3, wrapper.size, wrapper.size), output_path, {opset})
"""
    script_path = repo_dir / "_export_depth_pro.py"
    script_path.write_text(script_content)
    env = {**os.environ, "PYTHONPATH": str((repo_dir / "src").resolve())}
    run_cmd([python_path, str(script_path)], cwd=repo_dir, env=env, verbose=verbose)
    return repo_dir / output_name
