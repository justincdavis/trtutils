# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from trtutils.download._tools import _640, handle_imgsz, run_cmd, run_download, run_uv_pip_install

# rows kept by the in-graph NMS (mmdeploy's keep_top_k)
_RTMO_MAX_DET = 50
# rows scoring under this are not decoded in the graph (mmpose's test_cfg score_thr)
_RTMO_SCORE_FLOOR = 0.1


def export_rtmo(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,  # noqa: ARG001
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # OpenMMLab ships RTMO as ready ONNX (mmdeploy onnx_sdk zips, opset 11, 640x640, NMS
    # inside), so there is nothing to export: download, check and patch the graph
    handle_imgsz(imgsz, _640, "RTMO", enforce=True)
    run_download(directory, config, python_path, no_cache=no_cache, verbose=verbose)
    archive = directory / Path(urlsplit(config["url"]).path).name
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != config["sha256"]:
        err_msg = f"{archive.name} has sha256 {digest}, expected {config['sha256']}"
        raise RuntimeError(err_msg)
    with zipfile.ZipFile(archive) as zf:
        (directory / "end2end.onnx").write_bytes(zf.read("end2end.onnx"))
    run_uv_pip_install(directory, bin_path.parent, "rtmo", no_cache=no_uv_cache, verbose=verbose)

    onnx_name = f"{model}.onnx"
    # 1. rtmo was trained on bgr images, a Gather flips the channels so the model takes
    #    rgb like the other models
    # 2. the in-graph NMS drops boxes scoring under 0.15, so an image without people
    #    selects none and TensorRT 8.6 fails at run time (division by 0 in the shape
    #    graph): NMS runs without a score threshold, and the rows the keypoint head
    #    decodes are those scoring at least _RTMO_SCORE_FLOOR plus one, never zero.
    #    lower scores never suppress higher ones and NMS selects by descending score, so
    #    capping it at K rows and lowering the floor leave every row >= 0.15 unchanged
    # 3. the in-graph NMS keeps a data-dependent number of rows (at most K), which the
    #    engine cannot allocate for: pad with K zero rows and keep the top K by score so
    #    both outputs have a static shape
    program = f"""
import numpy as np
import onnx
from onnx import helper, numpy_helper

K = {_RTMO_MAX_DET}
m = onnx.load("end2end.onnx")
g = m.graph

def const(name, value):
    g.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.int64), name))

for node in g.node:
    for names in (node.input, node.output):
        for i, name in enumerate(names):
            if name == "input":
                names[i] = "input_bgr"
            elif name in ("dets", "keypoints"):
                names[i] = name + "_dyn"
const("rgb_to_bgr", [2, 1, 0])
g.node.insert(0, helper.make_node("Gather", ["input", "rgb_to_bgr"], ["input_bgr"], axis=1))

nodes = list(g.node)
producer = {{out: node for node in nodes for out in node.output}}
(nms,) = [node for node in nodes if node.op_type == "NonMaxSuppression"]
const("nms_max_out", [K])
g.initializer.append(numpy_helper.from_array(np.array([-1.0], np.float32), "nms_score_thres"))
nms.input[2] = "nms_max_out"
nms.input[4] = "nms_score_thres"
# mmdeploy keeps the top min(K, rows + 1) of the NMS rows plus one zero row by score:
# count only the rows scoring at least SCORE_FLOOR, so the keypoint head after it
# decodes those (at least one row) instead of K
keep = next(n for n in nodes[nodes.index(nms):] if n.op_type == "TopK")
where = producer[producer[keep.input[1]].input[0]]
count = producer[producer[where.input[0]].input[1]]
assert (where.op_type, count.op_type) == ("Where", "Add")
g.initializer.extend([
    numpy_helper.from_array(np.array({_RTMO_SCORE_FLOOR}, np.float32), "keep_floor"),
    numpy_helper.from_array(np.array(1.0, np.float32), "keep_one"),
])
count_nodes = [
    helper.make_node("Less", [keep.input[0], "keep_floor"], ["keep_low"]),
    helper.make_node("Not", ["keep_low"], ["keep_high"]),
    helper.make_node("Cast", ["keep_high"], ["keep_high_f"], to=onnx.TensorProto.FLOAT),
    helper.make_node("ReduceSum", ["keep_high_f"], ["keep_per_image"], axes=[1], keepdims=0),
    helper.make_node("ReduceMax", ["keep_per_image"], ["keep_rows"], axes=[0], keepdims=0),
    helper.make_node("Add", ["keep_rows", "keep_one"], ["keep_rows_1"]),
    helper.make_node("Cast", ["keep_rows_1"], ["keep_count"], to=onnx.TensorProto.INT64),
]
at = nodes.index(count)
for i, node in enumerate(count_nodes):
    g.node.insert(at + i, node)
count.input[1] = "keep_count"

num_kpts = 17
const("pad_dets", [0, 0, 0, 0, K, 0])
const("pad_kpts", [0, 0, 0, 0, 0, K, 0, 0])
const("score_col", 4)
const("top_k", [K])
const("dets_cols", [1, 1, 5])
const("kpts_cols", [1, 1, num_kpts, 3])
g.node.extend([
    helper.make_node("Pad", ["dets_dyn", "pad_dets"], ["dets_pad"]),
    helper.make_node("Pad", ["keypoints_dyn", "pad_kpts"], ["kpts_pad"]),
    helper.make_node("Gather", ["dets_pad", "score_col"], ["scores_pad"], axis=2),
    helper.make_node("TopK", ["scores_pad", "top_k"], ["top_scores", "top_idx"], axis=1),
    helper.make_node("Unsqueeze", ["top_idx"], ["idx_dets_1"], axes=[2]),
    helper.make_node("Expand", ["idx_dets_1", "dets_cols"], ["idx_dets"]),
    helper.make_node("GatherElements", ["dets_pad", "idx_dets"], ["dets"], axis=1),
    helper.make_node("Unsqueeze", ["top_idx"], ["idx_kpts_1"], axes=[2, 3]),
    helper.make_node("Expand", ["idx_kpts_1", "kpts_cols"], ["idx_kpts"]),
    helper.make_node("GatherElements", ["kpts_pad", "idx_kpts"], ["keypoints"], axis=1),
])
batch = g.output[0].type.tensor_type.shape.dim[0].dim_param
shapes = {{"dets": [batch, K, 5], "keypoints": [batch, K, num_kpts, 3]}}
for out in g.output:
    out.CopyFrom(helper.make_tensor_value_info(out.name, onnx.TensorProto.FLOAT, shapes[out.name]))
onnx.checker.check_model(m)
onnx.save(m, "{onnx_name}")
"""
    run_cmd([python_path, "-c", program], cwd=directory, verbose=verbose)
    return directory / onnx_name
