from __future__ import annotations

import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return image


def preprocess_image(image_bgr: np.ndarray, input_size: int, task: str) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ✅ FIX: Removed CLAHE (to match training)
    if task == "tb":
        image_rgb = image_rgb  # keep as-is

    resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)

    # TB model already has Rescaling(1/255) inside model
    if task == "tb":
        prepared = resized.astype(np.float32)
    else:
        prepared = resized.astype(np.float32) / 255.0

    return np.expand_dims(prepared, axis=0)


def simple_brightness_heuristic(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0)


def validate_tb_xray_image(image_bgr: np.ndarray, min_side: int = 224) -> tuple[bool, str]:
    height, width = image_bgr.shape[:2]
    if height < min_side or width < min_side:
        return False, f"Image resolution too small for TB screening (min {min_side}x{min_side})."

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())

    # ⚠️ OPTIONAL: slightly relaxed threshold (better real-world usability)
    if sat_mean > 60.0:
        return False, "Image appears too colorful to be a chest X-ray."

    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    channel_delta = (np.abs(b - g) + np.abs(g - r) + np.abs(b - r)) / 3.0

    if float(channel_delta.mean()) > 15.0:
        return False, "Image channels are inconsistent with grayscale X-ray data."

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    contrast_std = float(gray.std())

    if contrast_std < 15.0:
        return False, "Image contrast is too low for a usable chest X-ray."

    brightness = float(gray.mean())

    if brightness < 15.0 or brightness > 240.0:
        return False, "Image brightness is out of expected chest X-ray range."

    return True, "ok"
