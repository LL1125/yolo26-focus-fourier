"""Visualize placeholder contour predictions from the FocusContourNet contour branch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.predictor import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the placeholder contour branch output.")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model" / "fcn_focus_contour.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "contour_vis.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(args.model_config, weights=args.weights)
    outputs = predictor.predict_image(args.image, return_raw=True)
    contour = outputs.get("contour_outputs")
    if contour is None:
        raise RuntimeError("Contour outputs are unavailable for the selected model.")

    points = contour["refined_contour"][0, 0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(points[:, 0], points[:, 1], marker="o", linewidth=1)
    ax.set_title("Placeholder contour prediction")
    ax.set_aspect("equal")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
