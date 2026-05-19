"""CLI entrypoint for end-to-end prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.predictor import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end prediction.")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model" / "fcn_base.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "predict")
    parser.add_argument("--name", type=str, default="exp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor(args.model_config, weights=args.weights, device=args.device)
    outputs = predictor.predict_image(
        args.image,
        return_raw=args.raw,
        conf=args.conf,
        save=args.save,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
    )

    print(
        f"image={outputs['image_path']} detections={outputs['num_detections']} "
        f"conf={args.conf:.2f}"
        + (f" save_dir={outputs['save_dir']}" if outputs.get("save_dir") else "")
    )
    for idx, det in enumerate(outputs["summary"], start=1):
        print(
            f"{idx}: class={det['class_name']} class_id={det['class_id']} "
            f"conf={det['confidence']:.3f} box={det['box']}"
        )
    if outputs.get("saved_image_path"):
        print(f"saved_image={outputs['saved_image_path']}")
    if outputs.get("saved_txt_path"):
        print(f"saved_txt={outputs['saved_txt_path']}")


if __name__ == "__main__":
    main()
