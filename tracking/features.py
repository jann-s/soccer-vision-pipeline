# features.py
"""
Feature utilities for color-based team clustering in soccer videos.

This module provides:
- Bounding-box clipping to image bounds.
- Extraction of an upper-body (jersey) patch from a person box.
- HSV histogram features with an explicit grass mask to reduce background leakage.
"""

import cv2
import numpy as np


def clip_box(x1, y1, x2, y2, w, h):
    """
    Clamp a bounding box to image bounds and validate its geometry.

    Args:
        x1 (int | float): Left coordinate of the box.
        y1 (int | float): Top coordinate of the box.
        x2 (int | float): Right coordinate of the box.
        y2 (int | float): Bottom coordinate of the box.
        w (int): Image width in pixels.
        h (int): Image height in pixels.

    Returns:
        tuple[int, int, int, int] | None:
            The clamped box (x1, y1, x2, y2) as integers, or None if invalid.

    Notes:
        - The function ensures coordinates lie within [0..w-1] x [0..h-1].
        - An invalid (empty) box is reported as None (x2 <= x1 or y2 <= y1).
    """
    # Clamp coordinates to image boundaries
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))

    # Validate non-empty geometry
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def jersey_patch(img, box, top_frac=0.45):
    """
    Extract the upper-body (jersey) patch from a person bounding box.

    Args:
        img (np.ndarray): Input BGR image of shape (H, W, 3).
        box (tuple[int, int, int, int]): Person box (x1, y1, x2, y2) in pixels.
        top_frac (float): Fraction of the box height taken from the top (default 0.45).

    Returns:
        np.ndarray | None:
            The BGR subimage containing the jersey region, or None if the box is invalid.

    Behavior:
        - The input box is clamped to the image bounds.
        - Only the top portion (e.g., top 45%) is returned, approximating the shirt area.

    Notes:
        This is a light-weight proxy for torso segmentation; some head/skin or background
        may remain when boxes are loose. Downstream grass masking mitigates background.
    """
    # Image dimensions
    h_img, w_img = img.shape[:2]

    # Clamp the box to image bounds
    x1, y1, x2, y2 = box
    clamped = clip_box(x1, y1, x2, y2, w_img, h_img)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped

    # Compute the top sub-box height
    h_box = y2 - y1
    y2_top = y1 + max(1, int(h_box * float(top_frac)))

    # Crop jersey region
    patch = img[y1:y2_top, x1:x2]
    if patch.size == 0:
        return None
    return patch


def hsv_hist_feature(patch_bgr, bins_h=24, bins_s=8):
    """
    Compute a jersey color feature using HSV histograms with a grass mask.

    Args:
        patch_bgr (np.ndarray): BGR jersey patch (H x W x 3).
        bins_h (int): Number of hue histogram bins (default 24, range 0..180).
        bins_s (int): Number of saturation histogram bins (default 8, range 0..256).

    Returns:
        tuple[np.ndarray, float, float]:
            feat (np.ndarray): Concatenated and L1-normalized histogram [H|S], dtype=float32.
            mean_s (float): Mean saturation over non-grass pixels.
            mean_v (float): Mean value (brightness) over non-grass pixels.

    Behavior:
        - Converts the patch to HSV.
        - Applies a heuristic grass mask (green, sufficiently saturated/bright).
        - Computes masked 1D histograms for hue and saturation, concatenates them,
          and L1-normalizes the result.
        - Returns masked mean S and V as auxiliary stats (used for referee tie-breaks).

    Notes:
        Grass mask (heuristic):
            H ∈ [35, 85], S ≥ 60, V ≥ 40 (OpenCV HSV ranges: H∈[0,180], S,V∈[0,255]).
        Mask logic keeps NON-grass pixels to emphasize jersey color.
    """
    # Convert to HSV and split channels
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Heuristic grass mask: mark typical green field pixels
    grass = (H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)

    # Keep non-grass pixels (255 = keep, 0 = drop)
    mask = (~grass).astype(np.uint8) * 255

    # Histograms over non-grass pixels
    h_hist = cv2.calcHist([H], [0], mask, [bins_h], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], mask, [bins_s], [0, 256]).flatten()

    # Concatenate and L1-normalize
    feat = np.concatenate([h_hist, s_hist]).astype(np.float32)
    total = float(feat.sum())
    if total > 0.0:
        feat /= total

    # Robust mean S and V (fallback to full patch if mask empty)
    if np.any(mask):
        mean_s = float(S[mask > 0].mean())
        mean_v = float(V[mask > 0].mean())
    else:
        mean_s = float(S.mean())
        mean_v = float(V.mean())

    return feat, mean_s, mean_v