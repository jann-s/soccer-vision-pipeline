# utils.py
"""
Utility functions for soccer video processing.

This module provides:
- A ball pointer renderer (triangle arrow + subtle circle).
- A robust median absolute deviation (MAD) based threshold helper.
- A helper to resolve tracker YAML configuration paths.
"""

import os
import numpy as np
import cv2


def draw_ball_pointer(overlay, x1, y1, x2, y2, color=(0, 255, 255)):
    """
    Draw a downward triangle above the ball bounding box and a subtle circle at its center.

    Inputs:
        overlay (np.ndarray): BGR image on which to draw (modified in-place).
        x1 (int): Left coordinate of the ball bbox.
        y1 (int): Top coordinate of the ball bbox.
        x2 (int): Right coordinate of the ball bbox.
        y2 (int): Bottom coordinate of the ball bbox.
        color (tuple[int, int, int]): BGR color for the pointer and circle (default yellow).

    Output:
        None

    Behavior:
        - Builds a small triangle above the bbox top edge, horizontally centered.
        - Scales triangle size with bbox width.
        - Clamps triangle within image bounds on the vertical axis.
        - Draws a thin black outline around the triangle.
        - Draws a subtle ring around the ball bbox center for emphasis.
    """
    # Compute bbox width and center x
    w = max(1, int(x2 - x1))
    cx = (x1 + x2) // 2

    # Triangle geometry (scaled with bbox width)
    base = max(10, int(0.9 * w))     # triangle base width
    height = max(8, int(0.8 * w))    # triangle height
    margin = max(4, int(0.2 * w))    # vertical gap above bbox top

    # Triangle vertices: left, right, tip (downward triangle)
    tip = (cx, max(0, y1 - margin))
    left = (cx - base // 2, max(0, y1 - margin - height))
    right = (cx + base // 2, max(0, y1 - margin - height))
    pts = np.array([left, right, tip], dtype=np.int32)

    # Filled triangle
    cv2.fillPoly(overlay, [pts], color)
    # Thin black outline for readability
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # Subtle ring around ball center
    cy = (y1 + y2) // 2
    cv2.circle(overlay, (cx, cy), max(4, w // 3), color, 2, cv2.LINE_AA)


def mad_threshold(distances, k=2.5):
    """
    Compute a robust threshold using the Median Absolute Deviation (MAD).

    Inputs:
        distances (array-like): Collection of distances (floats).
        k (float): Scale factor applied to MAD (default 2.5).

    Output:
        float: Threshold value defined as median + k * 1.4826 * MAD.
               Returns +inf if the input is empty.

    Notes:
        - MAD = median(|x - median(x)|).
        - The constant 1.4826 makes MAD a consistent estimator of standard deviation
          under normality.
    """
    d = np.asarray(distances, dtype=np.float32)
    if d.size == 0:
        return float("inf")

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6  # small epsilon to avoid exact zero
    return med + k * 1.4826 * mad


def tracker_cfg_path(args_tracker, custom_cfg):
    """
    Resolve the tracker YAML configuration path.

    Inputs:
        args_tracker (str): Tracker key, e.g., "bytetrack" or "botsort".
        custom_cfg (str): Optional path to a user-provided YAML configuration file.

    Output:
        str: Effective YAML path. If 'custom_cfg' exists on disk, it is returned.
             Otherwise, "<args_tracker>.yaml" is returned (e.g., "bytetrack.yaml").

    Behavior:
        - Prefers a user-specified configuration when available.
        - Falls back to the default config name based on the tracker key.
    """
    # Use user-provided config if it exists
    if custom_cfg and os.path.isfile(custom_cfg):
        return custom_cfg

    # Fall back to "<tracker>.yaml"
    return f"{args_tracker}.yaml"