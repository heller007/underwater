"""Detection package."""

from .train import load_detector_config, predict_split, train_yolo

__all__ = ["load_detector_config", "train_yolo", "predict_split"]
