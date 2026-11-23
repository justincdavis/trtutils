# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.download._tools import (
    git_clone,
    normalize_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path


def export_fastreid(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | tuple[int, int] | None = None,
    *,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # resnet / resnest backbones end in global pooling, any (height, width) works
    size = normalize_imgsz(config["imgsz"] if imgsz is None else imgsz)
    height, width = (size, size) if isinstance(size, int) else size
    dynamic_axes = (
        '{"input": {0: "batch_size"}, "embedding": {0: "batch_size"}}' if dynamic else "None"
    )
    # v0.1.1 is the release the model zoo weights were published with: later commits
    # renamed the head layers and dropped the non-local blocks of the ResNeSt backbone
    git_clone(
        "https://github.com/JDAI-CV/fast-reid",
        directory,
        "64861ea6c85122bbf8577c10bd7fc6e551495b3a",
        no_cache=no_cache,
        verbose=verbose,
    )
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "fastreid",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(directory, config, python_path, no_cache=no_cache, verbose=verbose)
    output_name = f"{model}.onnx"
    script_content = f"""\
import sys

import onnx
import onnxslim
import torch

sys.path.insert(0, "fast-reid")
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model

output_path = "{output_name}"

cfg = get_cfg()
cfg.merge_from_file("fast-reid/configs/{config["config"]}")
cfg.MODEL.BACKBONE.PRETRAIN = False
cfg.MODEL.DEVICE = "cpu"
# the resnest builder indexes its tables by int depth, the configs say "50x"
if cfg.MODEL.BACKBONE.NAME == "build_resnest_backbone":
    cfg.MODEL.BACKBONE.DEPTH = int(str(cfg.MODEL.BACKBONE.DEPTH).rstrip("x"))
state = torch.load(
    "{config["name"]}.{config["extension"]}", map_location="cpu", weights_only=True
)["model"]
# the classifier is sized for the training identities; eval mode returns the
# feature after the bn neck, so the exported graph does not contain it
cfg.MODEL.HEADS.NUM_CLASSES = state["heads.classifier.weight"].shape[0]
model = build_model(cfg)
model.load_state_dict(state, strict=True)
model.eval()

# fastreid normalizes 0-255 rgb in the network with the imagenet mean / std times 255;
# the export starts after that step, so it takes imagenet-normalized 0-1 input
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
torch.testing.assert_close(model.pixel_mean, imagenet_mean)
torch.testing.assert_close(model.pixel_std, imagenet_std)


class Embedding(torch.nn.Module):
    def __init__(self, reid):
        super().__init__()
        self.backbone = reid.backbone
        self.heads = reid.heads

    def forward(self, x):
        return self.heads(self.backbone(x))


# exports never need gradients
torch.set_grad_enabled(False)
dummy_input = torch.randn({batch}, 3, {height}, {width})
torch.onnx.export(
    Embedding(model),
    dummy_input,
    output_path,
    dynamo=False,
    opset_version={opset},
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={dynamic_axes},
)

slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    script_path = directory / "_export_fastreid.py"
    script_path.write_text(script_content)
    run_cmd(
        [python_path, str(script_path)],
        cwd=directory,
        verbose=verbose,
    )
    return directory / output_name
