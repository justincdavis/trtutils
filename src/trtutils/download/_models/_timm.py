# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.download._tools import handle_imgsz, run_cmd, run_uv_pip_install

if TYPE_CHECKING:
    from pathlib import Path


def export_timm_classifier(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,  # noqa: ARG001
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # vision transformers take any multiple of the patch size, timm resamples
    # the position embeddings of the pretrained weights to the requested size
    imgsz = handle_imgsz(imgsz, int(config["imgsz"]), model, adjust_div=int(config["patch"]))
    dynamic_axes = '{"input": {0: "batch_size"}, "output": {0: "batch_size"}}' if dynamic else "None"
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "timm_classifier",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    output_name = f"{model}.onnx"
    script_content = f"""\
import onnx
import onnxslim
import timm
import torch

imgsz = {imgsz}
output_path = "{output_name}"

# distilled models average their class and distillation heads in eval mode
model = timm.create_model("{config["name"]}", pretrained=True, img_size=imgsz)
model.eval()
# exports never need gradients
torch.set_grad_enabled(False)
dummy_input = torch.randn({batch}, 3, imgsz, imgsz)
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version={opset},
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={dynamic_axes},
)

slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    script_path = directory / "_export_timm_classifier.py"
    script_path.write_text(script_content)
    run_cmd(
        [python_path, str(script_path)],
        cwd=directory,
        verbose=verbose,
    )
    return directory / output_name


def export_timm_features(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,  # noqa: ARG001
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # vision transformers take any multiple of the patch size, the timm dinov3
    # models resample their position embeddings (rope) to the input size
    imgsz = handle_imgsz(imgsz, int(config["imgsz"]), model, adjust_div=int(config["patch"]))
    dynamic_axes = (
        '{"input": {0: "batch_size"}, "embedding": {0: "batch_size"}, '
        '"feature_map": {0: "batch_size"}}'
        if dynamic
        else "None"
    )
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "timm_classifier",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    output_name = f"{model}.onnx"
    script_content = f"""\
import onnx
import onnxslim
import timm
import torch

imgsz = {imgsz}
output_path = "{output_name}"


class ViTFeatures(torch.nn.Module):
    # class token and patch token map, both after the final norm
    def __init__(self, model):
        super().__init__()
        self.model = model
        # class + register tokens
        self.num_prefix_tokens = model.num_prefix_tokens
        self.grid = tuple(model.patch_embed.dynamic_feat_size((imgsz, imgsz)))

    def forward(self, x):
        tokens = self.model.forward_features(x)
        embedding = tokens[:, 0]
        patches = tokens[:, self.num_prefix_tokens :]
        h, w = self.grid
        feature_map = patches.reshape(-1, h, w, patches.shape[-1]).permute(0, 3, 1, 2)
        return embedding, feature_map


# the input size is not passed, timm builds dinov3 with a dynamic image size
model = timm.create_model("{config["name"]}", pretrained=True)
model.eval()
wrapper = ViTFeatures(model).eval()
# exports never need gradients
torch.set_grad_enabled(False)
dummy_input = torch.randn({batch}, 3, imgsz, imgsz)
# the dynamo exporter fails on a dynamic batch at opset 17
torch.onnx.export(
    wrapper,
    dummy_input,
    output_path,
    dynamo=False,
    opset_version={opset},
    input_names=["input"],
    output_names=["embedding", "feature_map"],
    dynamic_axes={dynamic_axes},
)

slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    script_path = directory / "_export_timm_features.py"
    script_path.write_text(script_content)
    run_cmd(
        [python_path, str(script_path)],
        cwd=directory,
        verbose=verbose,
    )
    return directory / output_name
