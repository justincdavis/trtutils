# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.download._tools import handle_imgsz, run_cmd, run_uv_pip_install

if TYPE_CHECKING:
    from pathlib import Path

# the four stages downsample by 4, 2, 2, 2
_STRIDE = 32


def export_segformer(
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
    imgsz = handle_imgsz(imgsz, int(config["imgsz"]), model, adjust_div=_STRIDE)
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "segformer",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    dynamic_axes = '{"input": {0: "batch_size"}, "logits": {0: "batch_size"}}' if dynamic else "None"
    # the decode head outputs logits at 1/4 of the input size; the argmax build hook upsamples
    script_content = f"""\
import torch
import onnx
import onnxslim
from transformers import SegformerForSemanticSegmentation
from transformers.models.segformer import modeling_segformer

imgsz = {imgsz}
output_path = "{model}.onnx"


# same math as upstream, but with 4D permutes instead of a transpose(1, 2) of a Linear
# output: TensorRT 10.15 (RTX 5080) miscomputes that pattern (Mix-FFN fc1, decode head
# projections) for any batch above 1 and returns garbage logits; batch 1 was unaffected
def dwconv_forward(self, hidden_states, height, width):
    batch_size, _, num_channels = hidden_states.shape
    hidden_states = hidden_states.reshape(batch_size, height, width, num_channels)
    hidden_states = self.dwconv(hidden_states.permute(0, 3, 1, 2))
    hidden_states = hidden_states.permute(0, 2, 3, 1)
    return hidden_states.reshape(batch_size, height * width, num_channels)


def decode_head_forward(self, encoder_hidden_states, **kwargs):
    size = encoder_hidden_states[0].shape[2:]
    states = []
    for state, linear_proj in zip(encoder_hidden_states, self.linear_projections):
        state = linear_proj.proj(state.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        states.append(
            torch.nn.functional.interpolate(state, size=size, mode="bilinear", align_corners=False)
        )
    hidden_states = self.linear_fuse(torch.cat(states[::-1], dim=1))
    return self.classifier(self.activation(self.batch_norm(hidden_states)))


modeling_segformer.SegformerDepthWiseConv.forward = dwconv_forward
modeling_segformer.SegformerDecodeHead.forward = decode_head_forward


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


model = Logits(SegformerForSemanticSegmentation.from_pretrained("{config["name"]}"))
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
    output_names=["logits"],
    dynamic_axes={dynamic_axes},
    dynamo=False,
)

slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    script_path = directory / "_export_segformer.py"
    script_path.write_text(script_content)
    run_cmd([python_path, str(script_path)], cwd=directory, verbose=verbose)
    return directory / f"{model}.onnx"
