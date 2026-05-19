"""Prediction helper for standalone inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from datasets.transforms import ResizeToTensor
from engine import build_model, load_checkpoint, load_model_config, resolve_device, resolve_project_path
from models.common.utils import load_yaml, xyxy_to_xywh


class Predictor:
    """Load a model and run end-to-end detection inference."""

    def __init__(self, model_config: str | Path | dict[str, Any], weights: str | Path | None = None, device: str = "auto") -> None:
        self.model_cfg = load_model_config(model_config)
        self.device = resolve_device(device)
        self.model = build_model(self.model_cfg).to(self.device)
        self.model.eval()
        self.checkpoint: dict[str, Any] = {}
        if weights is not None:
            self.checkpoint = load_checkpoint(weights, self.model, optimizer=None, map_location=self.device)
        self.transform = ResizeToTensor(int(self.model_cfg.get("img_size", 640) or 640))
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> dict[int, str]:
        names = self.model_cfg.get("names")
        if names is None:
            train_cfg = self.checkpoint.get("train_cfg", {}) if isinstance(self.checkpoint, dict) else {}
            data_cfg_path = train_cfg.get("data_config")
            if data_cfg_path:
                data_cfg = load_yaml(resolve_project_path(data_cfg_path))
                names = data_cfg.get("names")
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}

    def _prepare_save_dir(self, project: str | Path, name: str) -> Path:
        project_path = resolve_project_path(project)
        project_path.mkdir(parents=True, exist_ok=True)
        candidate = project_path / name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix = 2
        while True:
            incremented = project_path / f"{name}{suffix}"
            if not incremented.exists():
                incremented.mkdir(parents=True, exist_ok=False)
                return incremented
            suffix += 1

    def _scale_detections_to_original(self, detections: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
        if detections.numel() == 0:
            return detections.reshape(0, 6)
        scaled = detections.clone()
        orig_w, orig_h = meta["original_size"]
        resized_w, resized_h = meta["resized_size"]
        scale_x = float(orig_w) / float(resized_w)
        scale_y = float(orig_h) / float(resized_h)
        scaled[:, [0, 2]] *= scale_x
        scaled[:, [1, 3]] *= scale_y
        scaled[:, [0, 2]] = scaled[:, [0, 2]].clamp(0, float(orig_w))
        scaled[:, [1, 3]] = scaled[:, [1, 3]].clamp(0, float(orig_h))
        return scaled

    def _filter_detections(self, detections: torch.Tensor, conf: float) -> torch.Tensor:
        if detections.numel() == 0:
            return detections.reshape(0, 6)
        keep = detections[:, 4] >= float(conf)
        return detections[keep]

    def _class_name(self, cls_id: int) -> str:
        return self.class_names.get(int(cls_id), str(int(cls_id)))

    def _build_summary(self, detections: torch.Tensor, limit: int = 10) -> list[dict[str, Any]]:
        summary = []
        for det in detections[:limit]:
            cls_id = int(det[5].item())
            summary.append(
                {
                    "class_id": cls_id,
                    "class_name": self._class_name(cls_id),
                    "confidence": float(det[4].item()),
                    "box": [round(float(v), 2) for v in det[:4].tolist()],
                }
            )
        return summary

    def _save_txt(self, detections: torch.Tensor, txt_path: Path, image_size: tuple[int, int]) -> None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = image_size
        lines: list[str] = []
        if detections.numel() > 0:
            boxes_xywh = xyxy_to_xywh(detections[:, :4])
            boxes_xywh[:, [0, 2]] /= float(width)
            boxes_xywh[:, [1, 3]] /= float(height)
            for box, det in zip(boxes_xywh, detections):
                cls_id = int(det[5].item())
                conf = float(det[4].item())
                lines.append(
                    f"{cls_id} {box[0].item():.6f} {box[1].item():.6f} {box[2].item():.6f} {box[3].item():.6f} {conf:.6f}"
                )
        txt_path.write_text("\n".join(lines), encoding="utf-8")

    def _draw_detections(self, image: Image.Image, detections: torch.Tensor) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            cls_id_int = int(cls_id)
            label = f"{self._class_name(cls_id_int)} {conf:.2f}"
            color = (255, 180, 0)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            text_bbox = draw.textbbox((x1, y1), label)
            text_bg = (x1, max(0, y1 - (text_bbox[3] - text_bbox[1]) - 6), x1 + (text_bbox[2] - text_bbox[0]) + 8, y1)
            draw.rectangle(text_bg, fill=color)
            draw.text((text_bg[0] + 4, text_bg[1] + 2), label, fill=(0, 0, 0))
        return canvas

    def predict_image(
        self,
        image_path: str | Path,
        return_raw: bool = False,
        conf: float = 0.25,
        save: bool = False,
        save_txt: bool = False,
        project: str | Path = "runs/predict",
        name: str = "exp",
    ) -> dict[str, Any]:
        """Run model inference for a single image and optionally save YOLO-style artifacts."""
        resolved_image_path = resolve_project_path(image_path)
        image = Image.open(resolved_image_path).convert("RGB")
        tensor, meta = self.transform(image)
        inputs = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model.forward_infer(inputs, return_raw=return_raw)

        detections = outputs["detections"][0].detach().cpu()
        detections = self._scale_detections_to_original(detections, meta)
        detections = self._filter_detections(detections, conf)
        summary = self._build_summary(detections)

        save_dir: Path | None = None
        saved_image_path: Path | None = None
        saved_txt_path: Path | None = None
        if save or save_txt:
            save_dir = self._prepare_save_dir(project, name)
            if save:
                rendered = self._draw_detections(image, detections)
                saved_image_path = save_dir / resolved_image_path.name
                rendered.save(saved_image_path)
            if save_txt:
                saved_txt_path = save_dir / "labels" / f"{resolved_image_path.stem}.txt"
                self._save_txt(detections, saved_txt_path, image.size)

        outputs["detections"] = detections.unsqueeze(0)
        outputs["meta"] = meta
        outputs["image_path"] = resolved_image_path
        outputs["num_detections"] = int(detections.shape[0])
        outputs["summary"] = summary
        outputs["save_dir"] = save_dir
        outputs["saved_image_path"] = saved_image_path
        outputs["saved_txt_path"] = saved_txt_path
        return outputs
