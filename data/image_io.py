"""
Image loading and saving shared by dataset generation, training, evaluation and inference.

Supported image modes (dataset_parameters["mode"] / CONFIG["image_type"]):

    "rgb"     3 channels, read from each frame's "file_path" (OpenCV order: B, G, R)
    "ir"      1 channel,  read from each frame's "ir_path"
    "rgb_ir"  4 channels, B, G, R from "file_path" stacked with IR from "ir_path"

Channels are always stored in this order, so a model trained in "rgb_ir" mode predicts
[B, G, R, IR] per pixel.
"""

import os

import cv2
import numpy as np


MODE_CHANNELS = {
    "rgb": 3,
    "ir": 1,
    "rgb_ir": 4,
}


def channels_for_mode(mode: str) -> int:
    """Number of image channels of an image mode."""
    mode = mode.lower()

    if mode not in MODE_CHANNELS:
        raise ValueError(f"Unknown image mode '{mode}'. Choose one of {sorted(MODE_CHANNELS)}.")

    return MODE_CHANNELS[mode]


def _read(path: str, flag: int) -> np.ndarray:
    image = cv2.imread(path, flag)

    if image is None:
        raise FileNotFoundError(f"[ERROR] Could not load image: {path}")

    return image


def load_frame_image(dataset_root: str, frame: dict, mode: str, resolution: int) -> np.ndarray:
    """
    Load one frame of a transforms_*.json in the given mode.

    Returns a uint8 array of shape [resolution, resolution, channels].
    """
    mode = mode.lower()
    channels_for_mode(mode)
    parts = []

    if mode in ("rgb", "rgb_ir"):
        if "file_path" not in frame:
            raise KeyError("[ERROR] Key 'file_path' not found in the frame metadata.")
        parts.append(_read(os.path.join(dataset_root, frame["file_path"]), cv2.IMREAD_COLOR))

    if mode in ("ir", "rgb_ir"):
        if "ir_path" not in frame:
            raise KeyError("[ERROR] Key 'ir_path' not found in the frame metadata.")
        parts.append(_read(os.path.join(dataset_root, frame["ir_path"]), cv2.IMREAD_GRAYSCALE))

    resized = []
    for part in parts:
        part = cv2.resize(part, (resolution, resolution))
        resized.append(part if part.ndim == 3 else part[:, :, np.newaxis])

    return np.concatenate(resized, axis=-1)


def split_channels(image: np.ndarray, mode: str) -> dict:
    """
    Split an [H, W, C] image into viewable images for saving or logging.

    Returns {"rgb": [H, W, 3] (BGR)} and/or {"ir": [H, W, 1]}, depending on the mode.
    """
    mode = mode.lower()

    if mode == "rgb":
        return {"rgb": image[..., :3]}

    if mode == "ir":
        return {"ir": image[..., :1]}

    return {"rgb": image[..., :3], "ir": image[..., 3:4]}
