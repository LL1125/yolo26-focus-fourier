"""Minimal standalone trainer for detect-only experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.detect_dataset import DetectDataset
from engine import build_model, load_checkpoint, load_model_config, resolve_device, resolve_project_path, save_checkpoint
from engine.validator_detect import ValidatorDetect
from models.common.utils import load_yaml, set_seed
from models.losses.detect_loss_e2e import DetectLossE2E


class TrainerDetect:
    """Train the baseline detect-only model without external frameworks."""

    def __init__(self, train_config: str | Path | dict[str, Any]) -> None:
        self.cfg = load_yaml(train_config) if not isinstance(train_config, dict) else train_config
        set_seed(int(self.cfg.get("seed", 42)))
        self.device = resolve_device(self.cfg.get("device", "auto"))
        self.output_dir = resolve_project_path(self.cfg.get("output_dir", "runs/detect"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_cfg = load_model_config(self.cfg["model_config"])
        self.data_cfg = load_yaml(resolve_project_path(self.cfg["data_config"]))
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
        if self.cfg.get("resume"):
            checkpoint = load_checkpoint(self.cfg["resume"], self.model, self.optimizer, map_location=self.device)
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
            self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
            self.best_epoch = int(checkpoint.get("best_epoch", self.best_epoch))

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

    def train(self) -> None:
        """Run the training loop."""
        train_loader, val_loader = self.build_dataloaders()
        print(f"num_train_images={len(train_loader.dataset)}")
        print(f"num_val_images={len(val_loader.dataset)}")

        epochs = int(self.cfg.get("epochs", 100))
        eval_every = int(self.cfg.get("eval_every", 1))

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
                images = batch["images"].to(self.device)
                targets = batch["targets"]
                self._move_targets_to_device(targets)
                outputs = self.model(images)
                loss_dict = self.criterion(outputs["detection_outputs"], targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict["loss"].backward()
                self.optimizer.step()
                running_loss += float(loss_dict["loss"].item())

            train_loss = running_loss / max(len(train_loader), 1)
            print(f"epoch={epoch} train_loss={train_loss:.4f}")

            metrics: dict[str, float] | None = None
            if (epoch + 1) % eval_every == 0:
                metrics = self.validator.validate(self.model, val_loader, self.criterion)
                val_loss = float(metrics["val_loss"])
                val_iou = float(metrics["val_mean_best_iou"])
                print(f"epoch={epoch} val_loss={val_loss:.4f} val_mean_best_iou={val_iou:.4f}")

                is_best = (val_iou > self.best_metric) or (
                    val_iou == self.best_metric and val_loss < self.best_val_loss
                )
                if is_best:
                    self.best_metric = val_iou
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    save_checkpoint(
                        self.output_dir / "best.pt",
                        self.model,
                        self.optimizer,
                        epoch,
                        extra=self._checkpoint_extra(epoch, train_loss, metrics),
                    )
                    print(
                        f"[best] updated -> epoch={epoch} "
                        f"val_mean_best_iou={self.best_metric:.4f} val_loss={self.best_val_loss:.4f}"
                    )

            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.optimizer,
                epoch,
                extra=self._checkpoint_extra(epoch, train_loss, metrics),
            )

        print(
            f"training finished | best_epoch={self.best_epoch} "
            f"best_val_mean_best_iou={self.best_metric:.4f} best_val_loss={self.best_val_loss:.4f}"
        )
