"""Standalone detect trainer with YOLO-style artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.detect_dataset import DetectDataset
from engine import build_model, load_checkpoint, load_model_config, resolve_device, resolve_project_path, save_checkpoint
from engine.validator_detect import ValidatorDetect
from models.common.utils import load_yaml, set_seed
from models.losses.detect_loss_e2e import DetectLossE2E


class TrainerDetect:
    """Train the baseline detect-only model without external frameworks."""

    RESULTS_HEADER = [
        "epoch",
        "train_loss",
        "val_loss",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "val_mean_best_iou",
        "best_so_far_map50_95",
        "learning_rate",
    ]

    def __init__(self, train_config: str | Path | dict[str, Any]) -> None:
        self.cfg = load_yaml(train_config) if not isinstance(train_config, dict) else train_config
        set_seed(int(self.cfg.get("seed", 42)))
        self.device = resolve_device(self.cfg.get("device", "auto"))

        requested_output_dir = resolve_project_path(self.cfg.get("output_dir", "runs/detect/train"))
        if requested_output_dir.name == "detect":
            requested_output_dir = requested_output_dir / "train"
        self.output_dir = requested_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.output_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.args_path = self.output_dir / "args.yaml"
        self.results_path = self.output_dir / "results.csv"
        self.results_png_path = self.output_dir / "results.png"

        self.model_cfg = load_model_config(self.cfg["model_config"])
        self.data_cfg = load_yaml(resolve_project_path(self.cfg["data_config"]))
        self.class_names = self._load_class_names()
        self.model = build_model(self.model_cfg).to(self.device)
        self.criterion = DetectLossE2E(num_classes=int(self.model_cfg["num_classes"]), loss_cfg=self.cfg.get("loss")).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.cfg.get("weight_decay", 5e-4)),
        )
        self.validator = ValidatorDetect(self.device)
        self.start_epoch = 0
        self.best_metric = float("-inf")
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self._saved_train_previews = 0
        if self.cfg.get("resume"):
            checkpoint = load_checkpoint(self.cfg["resume"], self.model, self.optimizer, map_location=self.device)
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
            self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
            self.best_epoch = int(checkpoint.get("best_epoch", self.best_epoch))

    def _load_class_names(self) -> dict[int, str]:
        names = self.data_cfg.get("names", self.model_cfg.get("names", {}))
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders."""
        train_set = DetectDataset(self.data_cfg, split="train", img_size=int(self.cfg.get("img_size", 640)))
        val_set = DetectDataset(self.data_cfg, split="val", img_size=int(self.cfg.get("img_size", 640)))
        batch_size = int(self.cfg.get("batch_size", 8))
        num_workers = int(self.cfg.get("num_workers", 4))
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=DetectDataset.collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=DetectDataset.collate_fn,
        )
        return train_loader, val_loader

    def _move_targets_to_device(self, targets: list[dict[str, Any]]) -> None:
        for target in targets:
            target["labels"] = target["labels"].to(self.device)
            target["boxes"] = target["boxes"].to(self.device)

    def _checkpoint_extra(self, epoch: int, train_loss: float, metrics: dict[str, float] | None) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "model_cfg": self.model_cfg,
            "train_cfg": self.cfg,
            "train_loss": float(train_loss),
            "best_metric": float(self.best_metric),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
            "epoch": int(epoch),
        }
        if metrics is not None:
            extra["val_metrics"] = {k: float(v) for k, v in metrics.items()}
        return extra

    def _write_args_file(self) -> None:
        payload = dict(self.cfg)
        payload["resolved_output_dir"] = str(self.output_dir)
        with self.args_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    def _ensure_results_file(self) -> None:
        if self.results_path.exists() and self.results_path.stat().st_size > 0:
            return
        with self.results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.RESULTS_HEADER)
            writer.writeheader()

    def _append_results_row(self, epoch: int, train_loss: float, metrics: dict[str, float] | None) -> None:
        metrics = metrics or {}
        val_loss = float(metrics.get("val_loss", float("nan")))
        precision = float(metrics.get("precision", float("nan")))
        recall = float(metrics.get("recall", float("nan")))
        map50 = float(metrics.get("map50", float("nan")))
        map50_95 = float(metrics.get("map50_95", float("nan")))
        val_iou = float(metrics.get("val_mean_best_iou", float("nan")))
        best_map = self.best_metric if self.best_metric != float("-inf") else float("nan")
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "val_mean_best_iou": val_iou,
            "best_so_far_map50_95": float(best_map),
            "learning_rate": learning_rate,
        }
        with self.results_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.RESULTS_HEADER)
            writer.writerow(row)

    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        array = image_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        array = (array * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    def _draw_boxes(self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor | None = None) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            cls_id = int(labels[idx].item()) if labels.numel() > idx else -1
            name = self.class_names.get(cls_id, str(cls_id))
            label = name
            if scores is not None and scores.numel() > idx:
                label = f"{name} {float(scores[idx].item()):.2f}"
            color = (255, 180, 0)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            text_bbox = draw.textbbox((x1, y1), label)
            text_bg = (x1, max(0, y1 - (text_bbox[3] - text_bbox[1]) - 6), x1 + (text_bbox[2] - text_bbox[0]) + 8, y1)
            draw.rectangle(text_bg, fill=color)
            draw.text((text_bg[0] + 4, text_bg[1] + 2), label, fill=(0, 0, 0))
        return canvas

    def _render_batch_grid(
        self,
        images: torch.Tensor,
        targets: list[dict[str, Any]],
        predictions: list[dict[str, torch.Tensor]] | None = None,
        max_items: int = 4,
    ) -> Image.Image:
        items = min(max_items, int(images.shape[0]))
        tiles: list[Image.Image] = []
        for idx in range(items):
            image = self._tensor_to_pil(images[idx])
            image = self._draw_boxes(image, targets[idx]["boxes"].detach().cpu(), targets[idx]["labels"].detach().cpu())
            if predictions is not None:
                pred = predictions[idx]
                image = self._draw_boxes(
                    image,
                    pred["boxes"].detach().cpu(),
                    pred["labels"].detach().cpu(),
                    pred["scores"].detach().cpu(),
                )
            tiles.append(image)

        if not tiles:
            return Image.new("RGB", (640, 640), color=(32, 32, 32))

        tile_w, tile_h = tiles[0].size
        cols = min(2, len(tiles))
        rows = int(np.ceil(len(tiles) / cols))
        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(24, 24, 24))
        for idx, tile in enumerate(tiles):
            row = idx // cols
            col = idx % cols
            canvas.paste(tile, (col * tile_w, row * tile_h))
        return canvas

    def _save_train_batch_preview(self, batch_idx: int, images: torch.Tensor, targets: list[dict[str, Any]]) -> None:
        if self._saved_train_previews >= 3:
            return
        preview = self._render_batch_grid(images, targets, predictions=None)
        preview_path = self.output_dir / f"train_batch{self._saved_train_previews}.jpg"
        preview.save(preview_path, quality=95)
        if self._saved_train_previews == 0:
            preview.save(self.output_dir / "labels.jpg", quality=95)
        self._saved_train_previews += 1

    def _prepare_prediction_entries(self, detections: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        prepared: list[dict[str, torch.Tensor]] = []
        for det in detections.detach().cpu():
            if det.numel() == 0:
                prepared.append(
                    {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.long),
                        "scores": torch.zeros((0,), dtype=torch.float32),
                    }
                )
                continue
            prepared.append(
                {
                    "boxes": det[:, :4],
                    "scores": det[:, 4],
                    "labels": det[:, 5].long(),
                }
            )
        return prepared

    def _save_validation_previews(self, model: torch.nn.Module, val_loader: DataLoader) -> None:
        batch = next(iter(val_loader))
        images = batch["images"].to(self.device)
        targets = batch["targets"]
        self._move_targets_to_device(targets)
        with torch.no_grad():
            infer = model.forward_infer(images)
        pred_entries = self._prepare_prediction_entries(infer["detections"])
        label_grid = self._render_batch_grid(images, targets, predictions=None)
        pred_grid = self._render_batch_grid(images, targets, predictions=pred_entries)
        label_grid.save(self.output_dir / "val_batch0_labels.jpg", quality=95)
        pred_grid.save(self.output_dir / "val_batch0_pred.jpg", quality=95)

    def _plot_results(self) -> None:
        if not self.results_path.exists():
            return
        data = np.genfromtxt(self.results_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if data.size == 0:
            return
        if data.ndim == 0:
            data = np.array([data], dtype=data.dtype)
        epochs = data["epoch"].astype(np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        ax.plot(epochs, data["train_loss"], label="train_loss")
        ax.plot(epochs, data["val_loss"], label="val_loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(epochs, data["precision"], label="precision")
        ax.plot(epochs, data["recall"], label="recall")
        ax.set_title("Precision / Recall")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(epochs, data["map50"], label="mAP50")
        ax.plot(epochs, data["map50_95"], label="mAP50-95")
        ax.set_title("mAP")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(epochs, data["val_mean_best_iou"], label="val_mean_best_iou")
        ax.plot(epochs, data["learning_rate"], label="lr")
        ax.set_title("IoU / LR")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.results_png_path, dpi=200)
        plt.close(fig)

    def train(self) -> None:
        """Run the training loop."""
        train_loader, val_loader = self.build_dataloaders()
        self._write_args_file()
        self._ensure_results_file()
        print(f"num_train_images={len(train_loader.dataset)}")
        print(f"num_val_images={len(val_loader.dataset)}")
        print(f"output_dir={self.output_dir}")

        epochs = int(self.cfg.get("epochs", 100))
        eval_every = int(self.cfg.get("eval_every", 1))

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"train {epoch}", leave=False)):
                images = batch["images"].to(self.device)
                targets = batch["targets"]
                self._move_targets_to_device(targets)
                if epoch == self.start_epoch and self._saved_train_previews < 3:
                    self._save_train_batch_preview(batch_idx, images, targets)
                outputs = self.model(images)
                loss_dict = self.criterion(outputs["detection_outputs"], targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict["loss"].backward()
                self.optimizer.step()
                running_loss += float(loss_dict["loss"].item())

            train_loss = running_loss / max(len(train_loader), 1)
            metrics: dict[str, float] | None = None
            val_loss = float("nan")
            precision = float("nan")
            recall = float("nan")
            map50 = float("nan")
            map50_95 = float("nan")
            val_iou = float("nan")
            if (epoch + 1) % eval_every == 0:
                metrics = self.validator.validate(self.model, val_loader, self.criterion)
                val_loss = float(metrics["val_loss"])
                precision = float(metrics["precision"])
                recall = float(metrics["recall"])
                map50 = float(metrics["map50"])
                map50_95 = float(metrics["map50_95"])
                val_iou = float(metrics["val_mean_best_iou"])

                is_best = (map50_95 > self.best_metric) or (
                    map50_95 == self.best_metric and val_loss < self.best_val_loss
                )
                if is_best:
                    self.best_metric = map50_95
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    save_checkpoint(
                        self.weights_dir / "best.pt",
                        self.model,
                        self.optimizer,
                        epoch,
                        extra=self._checkpoint_extra(epoch, train_loss, metrics),
                    )
                    print(
                        f"[best] updated -> epoch={epoch} "
                        f"map50_95={self.best_metric:.4f} val_loss={self.best_val_loss:.4f}"
                    )
                self._save_validation_previews(self.model, val_loader)

            self._append_results_row(epoch, train_loss, metrics)
            self._plot_results()
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            best_display = self.best_metric if self.best_metric != float("-inf") else float("nan")
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"P={precision:.4f} R={recall:.4f} mAP50={map50:.4f} mAP50_95={map50_95:.4f} "
                f"val_mean_best_iou={val_iou:.4f} best_mAP50_95={best_display:.4f} lr={current_lr:.6f}"
            )

            save_checkpoint(
                self.weights_dir / "last.pt",
                self.model,
                self.optimizer,
                epoch,
                extra=self._checkpoint_extra(epoch, train_loss, metrics),
            )

        print(
            f"training finished | best_epoch={self.best_epoch} "
            f"best_map50_95={self.best_metric:.4f} "
            f"best_val_loss={self.best_val_loss:.4f} output_dir={self.output_dir}"
        )
