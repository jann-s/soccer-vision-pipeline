# tracking/viz.py
"""
Visualization utilities for soccer tracking.

This module provides:
- Foot anchor computation (center-bottom of a bounding box).
- Text rendering with a simple background shadow for readability.
- Color desaturation helper.
- Drawing of open foot rings for players (TEAM1/TEAM2/UNK) and an arrow marker for the ball.
- Small helpers for robust integer parsing and class-name lookup.

Notes:
- The open rings are drawn as two arc segments with a gap on top, so the ring
  visually "frames" the player's legs without covering them completely.
- The ball is highlighted with a triangle pointer and a subtle circle (see utils.draw_ball_pointer).
"""

import cv2
import numpy as np

from .utils import draw_ball_pointer


def foot_anchor(x1, y1, x2, y2, img_h, offset=2):
    """
    Compute a foot anchor point slightly below a bounding box.

    Inputs:
        x1 (int): Left of the box.
        y1 (int): Top of the box.
        x2 (int): Right of the box.
        y2 (int): Bottom of the box.
        img_h (int): Image height (used to clamp to bottom of the frame).
        offset (int): Vertical pixel offset below the box (default 2).

    Output:
        tuple[int, int]: (cx, fy) where cx is the horizontal center of the box,
                         and fy is slightly below the bottom edge (clamped).
    """
    # Horizontal center of the box
    cx = int((x1 + x2) / 2)

    # Foot y coordinate (slightly below the bottom edge), clamped to image bounds
    fy = min(img_h - 1, int(y2) + int(offset))
    return cx, fy


def put_text_with_bg(img, text, org, color=(255, 255, 255),
                     font=cv2.FONT_HERSHEY_SIMPLEX, fs=0.6, thick=2):
    """
    Draw readable text with a simple dark shadow background.

    Inputs:
        img (np.ndarray): BGR image to draw on (modified in-place).
        text (str): Text string to draw.
        org (tuple[int, int]): Bottom-left corner of the text (x, y).
        color (tuple[int, int, int]): Text color in BGR (default white).
        font (int): OpenCV font (default cv2.FONT_HERSHEY_SIMPLEX).
        fs (float): Font scale.
        thick (int): Text thickness.

    Output:
        None

    Behavior:
        - Draw a 1px black shadow first, then the colored text on top.
    """
    # Shadow for better legibility
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), font, fs, (0, 0, 0), thick, cv2.LINE_AA)
    # Foreground text
    cv2.putText(img, text, org, font, fs, color, thick, cv2.LINE_AA)


def _desaturate_bgr(color, desat=0.35, base=180):
    """
    Lightly desaturate and brighten a BGR color by mixing with a gray base.

    Inputs:
        color (tuple[int, int, int]): Original color in BGR.
        desat (float): Mix factor with the gray base, in [0..1] (default 0.35).
        base (int): Gray base value (0..255) used for mixing (default 180).

    Output:
        tuple[int, int, int]: Adjusted BGR color.

    Behavior:
        - Each channel is mixed: new = color*(1-desat) + base*desat.
        - Clamps to [0..255].
    """
    b, g, r = color
    b_new = int(max(0, min(255, b * (1.0 - desat) + base * desat)))
    g_new = int(max(0, min(255, g * (1.0 - desat) + base * desat)))
    r_new = int(max(0, min(255, r * (1.0 - desat) + base * desat)))
    return (b_new, g_new, r_new)


def draw_team_rings(
    img, xyxy, ids, cls, role_colors, tid_to_teamnum,
    ball_cls_id=None, mark_ball=True,
    include_unknown=True,
    history=None,
    ring_scale=1.25,      # larger rings
    open_gap_deg=80,      # gap on top in degrees (~60–100 is good)
    thickness=2,          # thinner line
    desaturate=0.35       # reduce saturation slightly
):
    """
    Draw open foot rings in team color and a pointer for the ball.

    Inputs:
        img (np.ndarray): BGR image to annotate.
        xyxy (np.ndarray): Array of shape (N, 4) with boxes [x1, y1, x2, y2].
        ids (np.ndarray | None): Array of shape (N,) with track ids or None.
        cls (np.ndarray | None): Array of shape (N,) with class ids or None.
        role_colors (dict[str, tuple[int,int,int]]): BGR colors per role.
        tid_to_teamnum (dict[int, tuple[str,int]] | None):
            Mapping tid -> (role, team_number). Only TEAM1/TEAM2 have numbers.
        ball_cls_id (int | None): Class id used for the ball.
        mark_ball (bool): If True, draw the ball pointer instead of a ring.
        include_unknown (bool): If True, draw rings for UNK players (no number).
        history (defaultdict[int, deque] | None): Per-id trail of points for drawing motion history.
        ring_scale (float): Global scaling factor for ring axes (visual tuning).
        open_gap_deg (int): Angular size of the top gap in degrees.
        thickness (int): Line thickness of the ring arcs.
        desaturate (float): Mix factor for desaturating team color.

    Output:
        np.ndarray: Annotated BGR image.

    Behavior:
        - If the detection is a ball and mark_ball=True, draw a pointer (triangle + circle).
        - For TEAM1/TEAM2, draw an open ring under the player plus a team-local number.
        - For UNK (if enabled), draw a gray ring without a number.
        - Append foot-point positions to 'history' and draw trails with arrows.
        - Blend the overlay for a soft visual appearance.
    """
    # If there are no boxes, return the image unchanged
    if xyxy is None or len(xyxy) == 0:
        return img

    # Prepare an overlay to allow soft blending
    h, w = img.shape[:2]
    overlay = img.copy()

    # Draw each detection
    N = len(xyxy)
    for i in range(N):
        # Extract tid and class safely
        tid = int(ids[i]) if ids is not None else -1
        cls_id = int(cls[i]) if cls is not None else -1

        # Extract box and validate geometry
        x1, y1, x2, y2 = map(int, xyxy[i])
        if x2 <= x1 or y2 <= y1:
            continue

        # Ball: draw pointer only (no ring)
        if mark_ball and (ball_cls_id is not None) and (cls_id == ball_cls_id):
            draw_ball_pointer(overlay, x1, y1, x2, y2, color=(0, 255, 255))
            continue

        # Team role and number lookup
        info = None
        if tid_to_teamnum is not None:
            info = tid_to_teamnum.get(tid, None)

        # Decide role and base color
        if info is None:
            # No team assignment or not visible as TEAM1/TEAM2
            if not include_unknown or tid < 0:
                # Skip drawing for unknown tracks when include_unknown=False
                continue
            role = "UNK"
            base_color = role_colors.get("UNK", (128, 128, 128))
            number = None
        else:
            # info is (role, team_number)
            role, number = info
            base_color = role_colors.get(role, (0, 255, 0))

        # Lightly desaturate the base color for a softer look
        color = _desaturate_bgr(base_color, desat=desaturate, base=180)

        # Compute foot anchor slightly below the box to "frame" the legs
        cx = int((x1 + x2) // 2)
        fy = min(h - 1, y2 + 3)

        # Choose ring size relative to the box height
        box_h = max(1, y2 - y1)
        r = max(6, int(0.11 * box_h))  # base radius (larger)
        axes = (int(ring_scale * 1.8 * r), int(ring_scale * 0.9 * r))

        # Open ring: create a top gap (opening towards the upper side of the player)
        # Angle convention (OpenCV): 0° = right (3 o'clock); angles increase clockwise.
        # We draw two arcs leaving a gap centered "above" the ring.
        gap = max(20, min(160, int(open_gap_deg)))

        # Two arc segments to create the top opening:
        # Segment 1: from (270 + gap/2) to 360
        # Segment 2: from 0 to (270 - gap/2)
        start1 = 270 + gap // 2
        end1 = 360
        start2 = 0
        end2 = 270 - gap // 2

        # Draw both arcs
        cv2.ellipse(overlay, (cx, fy), axes, 0, start1, end1, color, thickness, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx, fy), axes, 0, start2, end2, color, thickness, cv2.LINE_AA)

        # Draw trail (history) from foot points
        if history is not None and tid >= 0:
            # Append current foot point
            history[tid].append((cx, fy))

            # Draw line segments between consecutive history points
            pts = list(history[tid])
            for j in range(1, len(pts)):
                cv2.line(overlay, pts[j - 1], pts[j], color, 2, cv2.LINE_AA)

            # Optional arrow on the last segment for motion direction
            if len(pts) >= 2:
                cv2.arrowedLine(overlay, pts[-2], pts[-1], color, 2, cv2.LINE_AA, tipLength=0.3)

        # Draw team-local number under the ring (TEAM1/TEAM2 only)
        if info is not None and number is not None:
            text = f"#{number}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Place text centered under the ellipse
            ty = min(h - 5, fy + axes[1] + th + 6)
            tx = max(5, cx - tw // 2)
            put_text_with_bg(overlay, text, (tx, ty), color, fs=0.6, thick=2)

    # Soft blend overlay onto original frame for a clean look
    blended = cv2.addWeighted(overlay, 0.85, img, 0.15, 0.0)
    return blended


def _safe_int(x, default=-1):
    """
    Safely convert a value to int, handling None and NaN.

    Inputs:
        x (Any): Value to convert.
        default (int): Fallback value if conversion fails (default -1).

    Output:
        int: Converted integer or default on failure.

    Behavior:
        - Returns 'default' if x is None or NaN.
        - Tries float->int conversion first; falls back to direct int().
    """
    import math as _math  # local import to avoid global dependency

    if x is None:
        return default

    try:
        xf = float(x)
        if _math.isnan(xf):
            return default
        return int(xf)
    except Exception:
        try:
            return int(x)
        except Exception:
            return default


def _name_lookup(names, cls_id):
    """
    Resolve a human-readable class name given an id and a name structure.

    Inputs:
        names (dict | list | tuple | None): Mapping or sequence of class names.
        cls_id (int): Class id.

    Output:
        str: Resolved class name or the numeric id as string.

    Behavior:
        - If 'names' is a dict, uses names[cls_id] with fallback to str(cls_id).
        - If 'names' is a list/tuple, uses bounds-checked indexing.
        - If 'names' is None, returns str(cls_id).
    """
    if names is None or cls_id < 0:
        return str(cls_id)

    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))

    if isinstance(names, (list, tuple)):
        if 0 <= cls_id < len(names):
            return str(names[cls_id])

    return str(cls_id)