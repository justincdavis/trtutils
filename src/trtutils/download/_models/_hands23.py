# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.download._tools import (
    git_clone,
    handle_imgsz,
    run_cmd,
    run_download,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path

_HANDS23_COMMIT = "1bc0f919ffa7a9f375e7e8042e1d26f3743d5819"
_DETECTRON2_COMMIT = "a2f4a8771ab77e8411c26b27f24f9489a28a2453"
_HANDS23_CONFIG = "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml"
_TOPK = 100


def export_hands23(
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
        LOG.warning("Hands23 uses detectron2, an Apache-2.0 licensed dependency")
        LOG.warning(
            "Hands23 export installs detectron2 from source (git+commit), which compiles "
            "C++/CUDA extensions and takes roughly 5-10 minutes; a C++ compiler is required"
        )
    imgsz = handle_imgsz(imgsz, 800, "Hands23", adjust_div=32)

    git_clone(
        "https://github.com/EvaCheng-cty/hands23_detector",
        directory,
        _HANDS23_COMMIT,
        no_cache=no_cache,
        verbose=verbose,
    )
    repo_dir = directory / "hands23_detector"

    run_uv_pip_install(
        repo_dir,
        bin_path.parent,
        "hands23",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    # detectron2 has no wheels and its setup.py needs torch already installed to build
    run_uv_pip_install(
        directory,
        bin_path.parent,
        None,
        # detectron2 imports torch in setup.py, so it must see the venv torch at build time
        packages=[
            "--no-build-isolation",
            f"git+https://github.com/facebookresearch/detectron2@{_DETECTRON2_COMMIT}",
        ],
        no_cache=no_uv_cache,
        verbose=verbose,
    )

    run_download(repo_dir, config, python_path, no_cache=no_cache, verbose=verbose)

    onnx_name = f"{model}.onnx"
    program = f"""
import math
import sys

sys.path.insert(0, r"{repo_dir}")

import onnx
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Boxes, ImageList

# importing this package registers hoRCNNROIHeads plus the z/h/t/g relation heads
from hodetector.modeling import roi_heads  # noqa: F401

S = {imgsz}
K = {_TOPK}
WEIGHTS = "{config["weights"]}"
OUT_PATH = "{onnx_name}"

cfg = get_cfg()
cfg.merge_from_file("{_HANDS23_CONFIG}")
cfg.MODEL.WEIGHTS = WEIGHTS
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.MASK_ON = False
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.TEST.DETECTIONS_PER_IMAGE = K
cfg.INPUT.MIN_SIZE_TEST = S
cfg.INPUT.MAX_SIZE_TEST = S

det_model = build_model(cfg)
DetectionCheckpointer(det_model).load(WEIGHTS)
det_model.eval()


def _sdf_scalar(hmin, hmax, val):
    if val < hmin:
        return hmin - val
    if val > hmax:
        return val - hmax
    return min(hmin - val, val - hmax)


def pair_features(f, boxes, classes):
    \"\"\"Vectorized re-derivation of hodetector.roi_heads.hoRCNNROIHeads.get_PF.\"\"\"
    k = f.shape[0]
    x1, y1, x2, y2 = boxes.unbind(-1)
    diag = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # hand -> first-object and first-object -> second-object are the only scored pairs
    hand_first = (classes[:, None] == 0) & (classes[None, :] == 1)
    obj_second = (classes[:, None] == 1) & (classes[None, :] == 2)
    mask = hand_first | obj_second

    def sdf(hmin, hmax, val):
        below = hmin - val
        above = val - hmax
        inside = torch.minimum(below, above)
        return torch.where(val < hmin, below, torch.where(val > hmax, above, inside))

    x1_i, x2_i, y1_i, y2_i = (t[:, None] for t in (x1, x2, y1, y2))
    x1_j, x2_j, y1_j, y2_j = (t[None, :] for t in (x1, x2, y1, y2))

    vx = (x1_j + x2_j - x1_i - x2_i) / 2
    vy = (y1_j + y2_j - y1_i - y2_i) / 2
    v_norm = torch.sqrt(vx**2 + vy**2)
    v_norm_safe = torch.where(v_norm > 0, v_norm, torch.ones_like(v_norm))

    diag_i = diag[:, None]
    diag_safe = torch.where(diag_i > 0, diag_i, torch.ones_like(diag_i))

    sdf_x1, sdf_x2 = sdf(x1_i, x2_i, x1_j), sdf(x1_i, x2_i, x2_j)
    sdf_y1, sdf_y2 = sdf(y1_i, y2_i, y1_j), sdf(y1_i, y2_i, y2_j)

    min_dist_x = torch.minimum(sdf_x1, sdf_x2) / diag_safe
    min_dist_y = torch.minimum(sdf_y1, sdf_y2) / diag_safe
    max_dist_x = torch.maximum(sdf_x1, sdf_x2) / diag_safe
    max_dist_y = torch.maximum(sdf_y1, sdf_y2) / diag_safe

    geom = torch.stack(
        [
            vx,
            vy,
            v_norm,
            min_dist_x,
            min_dist_y,
            max_dist_x,
            max_dist_y,
            vx / v_norm_safe,
            vy / v_norm_safe,
        ],
        dim=-1,
    )
    geom = geom * mask[..., None].to(geom.dtype)

    feat_i = f[:, None, :].expand(k, k, -1)
    feat_j = f[None, :, :].expand(k, k, -1)
    return torch.cat([feat_i, feat_j, geom], dim=-1)


def _self_check():
    # one hand-computed pair, checked against the scalar get_PF math verbatim
    hand_box = torch.tensor([10.0, 10.0, 30.0, 50.0])
    obj_box = torch.tensor([40.0, 20.0, 80.0, 60.0])
    boxes = torch.stack([hand_box, obj_box])
    classes = torch.tensor([0, 1])
    pf = pair_features(torch.zeros(2, 1024), boxes, classes)

    x1, y1, x2, y2 = hand_box.tolist()
    X1, Y1, X2, Y2 = obj_box.tolist()
    diag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    vx = (X1 + X2 - x1 - x2) / 2
    vy = (Y1 + Y2 - y1 - y2) / 2
    v_norm = math.sqrt(vx**2 + vy**2)
    min_dist_x = min(_sdf_scalar(x1, x2, X1), _sdf_scalar(x1, x2, X2)) / diag
    min_dist_y = min(_sdf_scalar(y1, y2, Y1), _sdf_scalar(y1, y2, Y2)) / diag
    max_dist_x = max(_sdf_scalar(x1, x2, X1), _sdf_scalar(x1, x2, X2)) / diag
    max_dist_y = max(_sdf_scalar(y1, y2, Y1), _sdf_scalar(y1, y2, Y2)) / diag
    expected = [vx, vy, v_norm, min_dist_x, min_dist_y, max_dist_x, max_dist_y, vx / v_norm, vy / v_norm]
    actual = pf[0, 1, 2048:].tolist()
    assert all(abs(a - b) < 1e-4 for a, b in zip(actual, expected)), (actual, expected)
    # pairs other than hand->first / first->second stay zeroed
    assert torch.all(pf[1, 0, 2048:] == 0)


_self_check()


class Wrapper(nn.Module):
    def __init__(self, det_model):
        super().__init__()
        self.det_model = det_model

    def forward(self, img):
        # img: (1,3,S,S) RGB 0-255 float -> BGR + detectron2 pixel normalization
        x = img.flip(1)
        x = (x - self.det_model.pixel_mean) / self.det_model.pixel_std
        images = ImageList(x, [(S, S)])

        feats = self.det_model.backbone(x)
        proposals, _ = self.det_model.proposal_generator(images, feats)

        roi_heads = self.det_model.roi_heads
        box_in = [feats[f] for f in roi_heads.box_in_features]
        box_feats = roi_heads.box_head(
            roi_heads.box_pooler(box_in, [p.proposal_boxes for p in proposals])
        )
        predictions = roi_heads.box_predictor(box_feats)
        pred_instances, _ = roi_heads.box_predictor.inference(predictions, proposals)
        inst = pred_instances[0]

        # pad/truncate to exactly K rows so the ONNX output shape is static
        pad_scores = F.pad(inst.scores, (0, K))
        scores, top_idx = pad_scores.topk(K)
        boxes = F.pad(inst.pred_boxes.tensor, (0, 0, 0, K))[top_idx]
        classes = F.pad(inst.pred_classes, (0, K))[top_idx]

        f = roi_heads.box_head(roi_heads.box_pooler(box_in, [Boxes(boxes)]))
        side = roi_heads.h_head(f).argmax(-1)
        touch = roi_heads.t_head(f).argmax(-1)

        pf = pair_features(f, boxes, classes)
        pair_probs = roi_heads.z_head(pf).softmax(-1)

        # first-object -> second-object links additionally require the first object's
        # touch prediction to be "tool, used" (2), matching hodetector's _inference_z
        # AND-gate; baked in-graph so the trtutils postprocessor only thresholds link prob
        gate = (classes[:, None] == 1) & (classes[None, :] == 2) & (touch[:, None] != 2)
        none_row = torch.zeros(5, dtype=pair_probs.dtype)
        none_row[0] = 1.0
        pair_probs = torch.where(gate[..., None], none_row, pair_probs)

        return (
            boxes[None],
            scores[None],
            classes[None].int(),
            pair_probs[None],
            side[None].int(),
        )


wrapper = Wrapper(det_model)
dummy = torch.zeros(1, 3, S, S)

# pip tensorrt wheels ship no libnvinfer_vc_plugin, which the parser's builtin
# RoiAlign importer requires just to report its VC library path (unused otherwise).
# emitting ROIAlign_TRT directly routes through the plugin-registry fallback importer
# instead, which needs no VC library. field names/values verified against TensorRT's
# plugin/roiAlignPlugin sources (release/10.15) and onnx-tensorrt's own RoiAlign
# importer in onnxOpImporters.cpp (main).
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args


@parse_args("v", "v", "f", "i", "i", "i", "b")
def _roi_align_trt(g, feats, rois, spatial_scale, out_h, out_w, sampling_ratio, aligned):
    # rois: (K,5) [batch_idx, x1, y1, x2, y2] -> split into batch indices + xyxy boxes
    zero, one, five = (g.op("Constant", value_t=torch.tensor([v], dtype=torch.int64)) for v in (0, 1, 5))
    idx = g.op("Slice", rois, zero, one, one)
    idx = g.op("Squeeze", idx, one)
    idx = g.op("Cast", idx, to_i=6)  # INT32
    boxes = g.op("Slice", rois, one, five, one)
    # onnx's checker rejects an unregistered op in the default domain; the "trt" domain
    # here is just to satisfy the checker -- the tensorrt parser looks up plugins by
    # op_type alone and ignores the onnx node domain
    return g.op(
        "trt::ROIAlign_TRT",
        feats,
        boxes,
        idx,
        plugin_version_s="2",
        plugin_namespace_s="",
        # detectron2/torchvision roi_align is average-pooling only
        mode_i=1,
        # aligned=True -> half_pixel (continuous coords), aligned=False -> output_half_pixel
        coordinate_transformation_mode_i=int(aligned),
        output_height_i=out_h,
        output_width_i=out_w,
        sampling_ratio_i=sampling_ratio,
        spatial_scale_f=spatial_scale,
    )


register_custom_op_symbolic("torchvision::roi_align", _roi_align_trt, {opset})

torch.onnx.export(
    wrapper,
    dummy,
    OUT_PATH,
    opset_version={opset},
    input_names=["input"],
    output_names=["boxes", "scores", "labels", "pair_probs", "side"],
    do_constant_folding=True,
    custom_opsets={{"trt": 1}},
)

# model is well under the 2GB protobuf limit; simplification is best-effort only
try:
    import onnxsim

    simplified, ok = onnxsim.simplify(OUT_PATH)
    if ok:
        onnx.save(simplified, OUT_PATH)
except Exception:
    pass
"""
    program_path = repo_dir / "_trtutils_export.py"
    program_path.write_text(program)
    run_cmd(
        [python_path, program_path],
        cwd=repo_dir,
        verbose=verbose,
    )

    return repo_dir / onnx_name
