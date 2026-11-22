# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from trtutils.download._tools import (
    git_clone,
    handle_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

# the checkpoint (weights/mobile_sam.pt) is part of the repository at this commit
_MOBILE_SAM_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
_SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"

# shared by both parts of every SAM-v1 style repo: each LayerNorm2d exports as one
# LayerNormalization node, whose hand-written (x - u) ** 2 otherwise overflows fp16
_SETUP = """
import onnxslim
import torch
from torch import nn
from torch.nn import functional as F

from {package} import sam_model_registry
from {package}.modeling import {norm_modules}


def _layer_norm_2d(self, x):
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
    return x.permute(0, 3, 1, 2)


for module in ({norm_modules},):
    module.LayerNorm2d.forward = _layer_norm_2d
torch.set_grad_enabled(False)
sam = sam_model_registry["{variant}"](checkpoint="{checkpoint}").eval()
"""

# normalized, padded (1, 3, S, S) image -> (1, 256, S / 16, S / 16) embeddings
_ENCODER = """
model = sam.image_encoder
inputs = (torch.randn(1, 3, {imgsz}, {imgsz}),)
names = (["images"], ["image_embeddings"])
dynamic_axes = None
"""

# prompt encoder + mask decoder without the in-graph upscale: B prompt sets of N points share
# one embedding, and all four low-res masks come out with their predicted IoU
_DECODER = """
class SamMaskDecoder(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.img_size = sam.image_encoder.img_size
        self.register_buffer("image_pe", sam.prompt_encoder.get_dense_pe().detach())

    def _embed_points(self, coords, labels):
        # segment_anything.utils.onnx.SamOnnxModel._embed_points, with the box labels 2 / 3
        pe = self.prompt_encoder
        coords = (coords + 0.5) / self.img_size
        emb = pe.pe_layer._pe_encoding(coords)
        labels = labels.unsqueeze(-1).expand_as(emb)
        emb = emb * (labels != -1)
        emb = emb + pe.not_a_point_embed.weight * (labels == -1)
        for i in range(pe.num_point_embeddings):
            emb = emb + pe.point_embeddings[i].weight * (labels == i)
        return emb

    def _embed_masks(self, mask_input, has_mask_input):
        pe = self.prompt_encoder
        emb = has_mask_input * pe.mask_downscaling(mask_input)
        return emb + (1 - has_mask_input) * pe.no_mask_embed.weight.reshape(1, -1, 1, 1)

    def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input):
        md = self.mask_decoder
        sparse = self._embed_points(point_coords, point_labels)
        dense = self._embed_masks(mask_input, has_mask_input)
        # MaskDecoder.predict_masks with expand instead of repeat_interleave
        tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0)
        tokens = tokens.unsqueeze(0).expand(sparse.shape[0], -1, -1)
        tokens = torch.cat((tokens, sparse), dim=1)
        src = image_embeddings.expand(tokens.shape[0], -1, -1, -1) + dense
        pos_src = self.image_pe.expand(tokens.shape[0], -1, -1, -1)
        hs, src = md.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + md.num_mask_tokens), :]
        # literal sizes keep the output dims static instead of Shape / Gather chains
        src = src.transpose(1, 2).reshape(-1, 256, 64, 64)
        up = md.output_upscaling(src)
        hyper_in = torch.stack(
            [md.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(md.num_mask_tokens)],
            dim=1,
        )
        masks = (hyper_in @ up.flatten(2)).reshape(-1, md.num_mask_tokens, 256, 256)
        return md.iou_prediction_head(iou_token_out), masks


model = SamMaskDecoder(sam).eval()
# traced at B=2, N=3 so neither dynamic axis is specialized to 1
inputs = (
    torch.randn(1, 256, 64, 64),
    torch.randint(0, {imgsz}, (2, 3, 2)).float(),
    torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, 1.0]]),
    torch.zeros(1, 1, 256, 256),
    torch.zeros(1),
)
names = (
    ["image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input"],
    ["iou_predictions", "low_res_masks"],
)
dynamic_axes = {{
    "point_coords": {{0: "num_prompts", 1: "num_points"}},
    "point_labels": {{0: "num_prompts", 1: "num_points"}},
    "iou_predictions": {{0: "num_prompts"}},
    "low_res_masks": {{0: "num_prompts"}},
}}
"""

_EXPORT = """
torch.onnx.export(
    model,
    inputs,
    "{output}",
    input_names=names[0],
    output_names=names[1],
    opset_version={opset},
    do_constant_folding=True,
    dynamic_axes=dynamic_axes,
    dynamo=False,
)
onnxslim.slim("{output}", "{output}")
"""


def _export_part(
    repo_dir: Path,
    config: dict[str, str],
    python_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    setup: dict[str, str],
    *,
    verbose: bool | None = None,
) -> Path:
    # the encoder or the decoder, see the config's "part"
    part = _ENCODER if config["part"] == "encoder" else _DECODER
    output = repo_dir.parent / f"{model}.onnx"
    program = _SETUP.format(**setup) + part.format(imgsz=imgsz)
    program += _EXPORT.format(output=output, opset=opset)
    run_cmd([python_path, "-c", program], cwd=repo_dir, verbose=verbose)
    return output


def export_mobile_sam(
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
    imgsz = handle_imgsz(imgsz, 1024, "MobileSAM", enforce=True)
    git_clone(
        "https://github.com/ChaoningZhang/MobileSAM",
        directory,
        _MOBILE_SAM_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    repo_dir = directory / "MobileSAM"
    run_uv_pip_install(
        repo_dir, bin_path.parent, "mobile_sam", no_cache=no_uv_cache, verbose=verbose
    )
    # the TinyViT encoder has its own LayerNorm2d class
    setup = {
        "package": "mobile_sam",
        "norm_modules": "common, tiny_vit_sam",
        "variant": "vit_t",
        "checkpoint": "weights/mobile_sam.pt",
    }
    return _export_part(repo_dir, config, python_path, model, opset, imgsz, setup, verbose=verbose)


def export_sam(
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
    imgsz = handle_imgsz(imgsz, 1024, "SAM", enforce=True)
    git_clone(
        "https://github.com/facebookresearch/segment-anything",
        directory,
        _SAM_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    repo_dir = directory / "segment-anything"
    run_uv_pip_install(repo_dir, bin_path.parent, "sam", no_cache=no_uv_cache, verbose=verbose)
    # the checkpoint of the config's "variant" (vit_b, vit_l)
    run_download(repo_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    setup = {
        "package": "segment_anything",
        "norm_modules": "common",
        "variant": config["variant"],
        "checkpoint": Path(urlsplit(config["url"]).path).name,
    }
    return _export_part(repo_dir, config, python_path, model, opset, imgsz, setup, verbose=verbose)
