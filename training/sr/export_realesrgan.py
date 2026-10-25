"""Export Real-ESRGAN realesr-general-x4v3 (BSD-3-Clause, xinntao/Real-ESRGAN) to ONNX for
Drishti's super-resolution slot (`models.sr_model`): NCHW float RGB in [0, 1], 4x output.

    pip install torch onnx
    curl -LO https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
    python training/sr/export_realesrgan.py realesr-general-x4v3.pth realesr_general_x4v3.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """The compact Real-ESRGAN generator: a plain conv/PReLU stack, pixel shuffle, nearest residual."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64, num_conv: int = 32, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="realesr-general-x4v3.pth")
    ap.add_argument("out", help="output .onnx")
    a = ap.parse_args()

    state = torch.load(a.weights, map_location="cpu", weights_only=True)
    state = state.get("params", state.get("params_ema", state))
    model = SRVGGNetCompact()
    model.load_state_dict(state)
    model.eval()
    x = torch.rand(1, 3, 32, 96)
    torch.onnx.export(model, x, a.out, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h4", 3: "w4"}}, dynamo=False)

    import onnxruntime as ort

    y = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    print(f"wrote {a.out}; max difference from PyTorch {float(np.abs(y - model(x).detach().numpy()).max()):.2e}")


if __name__ == "__main__":
    main()
