"""Export the FocusContourNet detector to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import build_model, load_checkpoint, load_model_config


class ONNXWrapper(torch.nn.Module):
    """Return only detection tensors so ONNX export stays simple."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_infer(x)["detections"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a detector to ONNX.")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model" / "fcn_base.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model.onnx")
    parser.add_argument("--img-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_model_config(args.model_config)
    model = build_model(model_cfg)
    if args.weights:
        load_checkpoint(args.weights, model, optimizer=None, map_location="cpu")
    model.eval()
    wrapper = ONNXWrapper(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, int(model_cfg.get("in_channels", 3)), args.img_size, args.img_size)
    torch.onnx.export(wrapper, dummy, args.output, opset_version=13, input_names=["images"], output_names=["detections"])
    print(args.output)


if __name__ == "__main__":
    main()
