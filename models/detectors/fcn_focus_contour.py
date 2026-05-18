"""FocusContourNet detector with focus plugins and a parallel contour branch."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbones.fcn_backbone import FCNBackbone
from models.heads.detect_head_e2e import DetectHeadE2E
from models.heads.fourier_contour_head import FourierContourHead
from models.plugins.feature_router import FeatureRouter
from models.plugins.focus_region_block import FocusRegionBlock


class FCNFocusContourDetector(nn.Module):
    """Detector that exposes both detection outputs and contour outputs."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        neck_channels = cfg["neck_channels"]
        router = FeatureRouter(
            {
                "backbone_late": FocusRegionBlock(cfg["backbone_channels"][3]),
                "neck_fusion": FocusRegionBlock(neck_channels[-2] if len(neck_channels) >= 2 else neck_channels[0]),
            }
        )
        self.backbone = FCNBackbone(cfg, plugin_router=router)
        self.detect_head = DetectHeadE2E(
            num_classes=int(cfg["num_classes"]),
            ch=neck_channels,
            strides=cfg["strides"],
            hidden_dim=int(cfg.get("head_channels", 128)),
            max_det=int(cfg.get("max_det", 300)),
        )
        fourier_cfg = cfg.get("fourier", {})
        self.contour_head = FourierContourHead(
            in_channels=neck_channels,
            hidden_dim=int(fourier_cfg.get("hidden_dim", 128)),
            num_coeffs=int(fourier_cfg.get("num_coeffs", 16)),
            num_points=int(fourier_cfg.get("num_points", 64)),
            boundary_head=bool(fourier_cfg.get("boundary_head", True)),
        )

    def forward_train(self, images: torch.Tensor) -> dict[str, Any]:
        """Return detection and contour outputs for joint training."""
        features = self.backbone(images)
        neck_features = features["features"]
        return {
            "feature_pyramid": features,
            "detection_outputs": self.detect_head.forward_train(neck_features),
            "contour_outputs": self.contour_head(neck_features),
        }

    def forward_infer(self, images: torch.Tensor, return_raw: bool = False, include_contour: bool = True) -> dict[str, Any]:
        """Run one-to-one detection inference and optionally emit contour predictions."""
        features = self.backbone(images)
        neck_features = features["features"]
        det_outputs = self.detect_head.forward_infer(neck_features)
        contour_outputs = self.contour_head(neck_features) if include_contour else None
        payload = {"detections": det_outputs["detections"], "contour_outputs": contour_outputs}
        if return_raw:
            payload["feature_pyramid"] = features
            payload["detection_branch_outputs"] = det_outputs["branch_outputs"]
        return payload

    def forward(self, images: torch.Tensor, return_raw: bool = False) -> dict[str, Any]:
        """Dispatch by module mode."""
        if self.training:
            return self.forward_train(images)
        return self.forward_infer(images, return_raw=return_raw)
