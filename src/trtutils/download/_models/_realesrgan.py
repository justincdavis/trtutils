# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from trtutils.download._tools import (
    normalize_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

# the two networks, as in basicsr's rrdbnet_arch.py (RRDBNet) and Real-ESRGAN's
# srvgg_arch.py (SRVGGNetCompact): defined here so the export needs neither package
# (basicsr fails to import with current torchvision); the weights load with strict=True
_ARCHS = """\
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


def pixel_unshuffle(x, factor):
    b, c, hh, hw = x.size()
    h, w = hh // factor, hw // factor
    x_view = x.view(b, c, h, factor, w, factor)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * factor * factor, h, w)


class RRDBNet(nn.Module):
    def __init__(self, scale=4, num_block=23, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        num_in_ch = 3 * {4: 1, 2: 4, 1: 16}[scale]
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_conv=32, upscale=4, num_feat=64):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
"""


def _hw(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    return (imgsz, imgsz) if isinstance(imgsz, int) else imgsz


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_realesrgan(
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
    height, width = _hw(normalize_imgsz(imgsz if imgsz is not None else config["imgsz"]))
    scale = int(config["scale"])
    # the x2 network pixel-unshuffles its input by 2
    divisor = {2: 2, 1: 4}.get(scale, 1)
    if height % divisor or width % divisor:
        err_msg = (
            f"{model} requires a height and width divisible by {divisor}, got {(height, width)}"
        )
        raise ValueError(err_msg)
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "realesrgan",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(directory, config, python_path, no_cache=no_cache, verbose=verbose)
    weights = directory / Path(urlsplit(config["url"]).path).name
    digest = _sha256(weights)
    if digest != config["sha256"]:
        err_msg = f"{weights.name} has sha256 {digest}, expected {config['sha256']}"
        raise RuntimeError(err_msg)

    network = (
        f"RRDBNet(scale={scale}, num_block={int(config['depth'])})"
        if config["arch"] == "rrdbnet"
        else f"SRVGGNetCompact(num_conv={int(config['depth'])}, upscale={scale})"
    )
    output_name = f"{model}.onnx"
    script_content = f"""\
{_ARCHS}

import onnx
import onnxslim

# exports never need gradients
torch.set_grad_enabled(False)
model = {network}
state = torch.load("{weights.name}", map_location="cpu", weights_only=True)
model.load_state_dict(state["params_ema"] if "params_ema" in state else state["params"], strict=True)
model.eval()
torch.onnx.export(
    model,
    torch.rand(1, 3, {height}, {width}),
    "{output_name}",
    opset_version={opset},
    input_names=["input"],
    output_names=["output"],
    dynamo=False,
)
onnx.save(onnxslim.slim("{output_name}"), "{output_name}")
"""
    script_path = directory / "_export_realesrgan.py"
    script_path.write_text(script_content)
    run_cmd([python_path, str(script_path)], cwd=directory, verbose=verbose)
    return directory / output_name
