"""Detector exports."""

from models.detectors.fcn_base import FCNBaseDetector
from models.detectors.fcn_focus_contour import FCNFocusContourDetector

__all__ = ["FCNBaseDetector", "FCNFocusContourDetector"]
