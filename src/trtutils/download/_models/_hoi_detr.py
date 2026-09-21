# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.download._tools import (
    git_clone,
    handle_batch,
    handle_dynamic,
    handle_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path

_HOI_DETR_COMMIT = "1b367292f3833afd64a204bd4d9d84519541d035"
_HOI_DETR_CONFIG = (
    "projects/configs/co_dino_vit/"
    "co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py"
)
_TOPK = 100


def export_hoi_detr(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    batch: int | None = None,
    *,
    dynamic: bool | None = None,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("HOI-DETR is a MIT licensed model, be aware of license restrictions")
        LOG.warning(
            "HOI-DETR checkpoint is ~5.86 GB, export needs roughly 16 GB of RAM to load "
            "and trace the ViT-L backbone"
        )
    imgsz = handle_imgsz(imgsz, 640, "HOI-DETR", adjust_div=32)
    handle_batch(batch, "HOI-DETR", supported=False)
    handle_dynamic(batch, "HOI-DETR", dynamic=dynamic, supported=False)

    git_clone(
        "https://github.com/AhmadDarKhalil/HOI-DETR",
        directory,
        _HOI_DETR_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    repo_dir = directory / "HOI-DETR"

    run_uv_pip_install(
        repo_dir,
        bin_path.parent,
        "hoi_detr",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    # the repo vendors mmdet 2.25.3, pip install -e . is its documented install
    run_uv_pip_install(
        directory,
        bin_path.parent,
        None,
        # the vendored mmdet setup.py imports torch, so build against the venv torch
        packages=["--no-build-isolation", "-e", str(repo_dir)],
        no_cache=no_uv_cache,
        verbose=verbose,
    )

    run_download(repo_dir, config, python_path, no_cache=no_cache, verbose=verbose)

    onnx_name = f"{model}.onnx"
    program = f"""
import onnx
import torch
from torch import nn

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

from projects import *  # noqa: F401,F403  registers CoDETR, CoDINOHeadWithInteraction, ViT, SFP

S = {imgsz}
K = {_TOPK}
WEIGHTS = "{config["weights"]}"
OUT_PATH = "{onnx_name}"

cfg = Config.fromfile("{_HOI_DETR_CONFIG}")
cfg.model.backbone.use_act_checkpoint = False
cfg.model.query_head.transformer.encoder.with_cp = 0
cfg.model.train_cfg = None
if "pretrained" in cfg.model:
    cfg.model.pretrained = None

det = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(det, WEIGHTS, map_location="cpu")
det.eval()


class Wrapper(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, img):
        b = img.shape[0]
        metas = [
            dict(
                img_shape=(S, S, 3),
                batch_input_shape=(S, S),
                pad_shape=(S, S, 3),
                scale_factor=1.0,
            )
        ] * b
        feats = self.det.extract_feat(img)
        outs, hs = self.det.query_head(feats, metas, return_hs=True)
        logits = outs[0][-1]  # (B,1500,3)
        boxes_cxcywh = outs[1][-1]  # (B,1500,4) normalized
        emb = hs[-1]  # (B,1500,256)

        prob = logits.sigmoid().flatten(1)  # (B,4500)
        scores, idx = prob.topk(K, dim=1)
        q = idx // 3
        labels = idx % 3

        gathered = torch.gather(boxes_cxcywh, 1, q.unsqueeze(-1).expand(-1, -1, 4))
        cx, cy, w, h = gathered.unbind(-1)
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * S

        e = torch.gather(emb, 1, q.unsqueeze(-1).expand(-1, -1, emb.shape[-1]))
        pairs = torch.cat(
            [e.unsqueeze(2).expand(-1, -1, K, -1), e.unsqueeze(1).expand(-1, K, -1, -1)],
            -1,
        )
        pair_probs = self.det.query_head.interaction_head.mlp(pairs).softmax(-1)  # (B,K,K,2)
        return boxes, scores, labels.to(torch.int32), pair_probs


wrapper = Wrapper(det)
dummy = torch.zeros(1, 3, S, S)

torch.onnx.export(
    wrapper,
    dummy,
    OUT_PATH,
    opset_version={opset},
    input_names=["input"],
    output_names=["boxes", "scores", "labels", "pair_probs"],
    dynamic_axes={{
        "input": {{0: "batch"}},
        "boxes": {{0: "batch"}},
        "scores": {{0: "batch"}},
        "labels": {{0: "batch"}},
        "pair_probs": {{0: "batch"}},
    }},
    do_constant_folding=True,
)

# the model exceeds the 2GB protobuf limit; torch.onnx.export may already have spilled
# weights to per-initializer external files next to OUT_PATH depending on version -- reload
# and collapse to a single external data file regardless of which path torch took.
# skip onnxsim here, it does not handle >2GB external-data models.
onnx_model = onnx.load(OUT_PATH, load_external_data=True)
onnx.save_model(
    onnx_model,
    OUT_PATH,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=OUT_PATH + ".data",
)
"""
    program_path = repo_dir / "_trtutils_export.py"
    program_path.write_text(program)
    run_cmd(
        [python_path, program_path],
        cwd=repo_dir,
        verbose=verbose,
    )

    return repo_dir / onnx_name
