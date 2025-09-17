# tracking/clustering.py
"""
Color-based team clustering for soccer tracking.

This module provides:
- Utility functions to crop jersey patches and compute HSV histogram features
  with an explicit grass mask (to reduce background leakage).
- A TeamClusterer that keeps per-track color features (EMA-smoothed),
  performs K-Means with K=3 (TEAM1, TEAM2, REF), maps the two team clusters
  to fixed roles via a previous-centers assignment (permutation protection),
  and assigns final roles for all active tracks (including BALL and UNK).
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
        w (int): Image width.
        h (int): Image height.

    Returns:
        tuple[int, int, int, int] | None: (x1, y1, x2, y2) clamped to [0..w-1]/[0..h-1],
        or None if the box is invalid (non-positive width/height).

    Behavior:
        - Ensures coordinates are within the image.
        - Returns None when x2 <= x1 or y2 <= y1.
    """
    # Clamp coordinates to image bounds
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))

    # Validate box geometry
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def jersey_patch(img, box, top_frac=0.45):
    """
    Extract the top fraction of a person box as a jersey patch.

    Args:
        img (np.ndarray): BGR image (H x W x 3).
        box (array-like): [x1, y1, x2, y2] in pixels.
        top_frac (float): Fraction of the box height to keep from the top.

    Returns:
        np.ndarray | None: BGR patch (jersey region) or None if invalid.

    Behavior:
        - Clamps the input box to the image.
        - Uses only the top portion (e.g., 45%) where jersey color dominates.
    """
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # Ensure the box is valid within the image
    box_ok = clip_box(x1, y1, x2, y2, w_img, h_img)
    if box_ok is None:
        return None
    x1, y1, x2, y2 = box_ok

    # Compute top sub-box (jersey region)
    h_box = y2 - y1
    y2_top = y1 + max(1, int(h_box * float(top_frac)))

    # Crop the jersey patch
    patch = img[y1:y2_top, x1:x2]
    if patch.size == 0:
        return None
    return patch


def hsv_hist_feature(patch_bgr, bins_h=24, bins_s=8):
    """
    Compute a color feature from a jersey patch using HSV histograms.

    Args:
        patch_bgr (np.ndarray): BGR image patch (jersey region).
        bins_h (int): Number of bins for Hue histogram (range 0..180).
        bins_s (int): Number of bins for Saturation histogram (range 0..256).

    Returns:
        tuple[np.ndarray, float, float]:
            - feat (np.ndarray, float32): Concatenated and L1-normalized [H-hist | S-hist].
            - mean_s (float): Mean saturation of non-grass pixels.
            - mean_v (float): Mean value (brightness) of non-grass pixels.

    Behavior:
        - Converts patch to HSV.
        - Builds a "grass" mask to exclude likely field pixels (green range).
        - Computes masked 1D histograms for H and S and concatenates them.
        - Normalizes the feature vector to sum to 1 (if non-zero).
        - Also returns masked mean S and V for later heuristics (e.g., referee tie-break).
    """
    # Convert to HSV
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Grass mask: exclude typical green field pixels
    grass = (H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)
    mask = (~grass).astype(np.uint8) * 255  # 255 = keep (non-grass), 0 = discard

    # Compute masked histograms for H and S
    h_hist = cv2.calcHist([H], [0], mask, [bins_h], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], mask, [bins_s], [0, 256]).flatten()

    # Concatenate and normalize
    feat = np.concatenate([h_hist, s_hist]).astype(np.float32)
    ssum = float(feat.sum())
    if ssum > 0.0:
        feat /= ssum

    # Mean S and V over non-grass pixels (fallback to full patch if mask empty)
    if np.any(mask):
        mean_v = float(V[mask > 0].mean())
        mean_s = float(S[mask > 0].mean())
    else:
        mean_v = float(V.mean())
        mean_s = float(S.mean())

    return feat, mean_s, mean_v


class TeamClusterer:
    """
    Color-based K-Means clustering (K=3) on jersey features for role assignment.

    Roles produced:
        - TEAM1, TEAM2, REF, BALL, UNK

    Stability mechanisms:
        - Grass-masked HSV histograms with per-track EMA smoothing
        - REF chosen as the smallest cluster (tie-break via S/V score)
        - Team clusters mapped to TEAM1/TEAM2 using previous centers (2x2 distance)
        - No additional temporal smoothing of cluster centers

    Attributes:
        bins_h (int): Hue bins for feature extraction.
        bins_s (int): Saturation bins for feature extraction.
        ema (float): EMA factor for per-track feature smoothing (alpha in [0..1]).
        track_feat (dict[int, np.ndarray]): EMA-smoothed feature per track id.
        track_stats (dict[int, dict]): Per-track stats {"s": mean_s, "v": mean_v}.
        prev_centers (list[np.ndarray] | None): Previous centers [team1, team2, ref].
        roles (dict[int, dict]): Assigned role per track id, {"role": str}.
        role_colors (dict[str, tuple[int,int,int]]): BGR colors for rendering per role.
    """

    def __init__(self, bins_h=24, bins_s=8, ema=0.6):
        """
        Initialize the clustering helper and internal state.

        Args:
            bins_h (int): Hue histogram bins.
            bins_s (int): Saturation histogram bins.
            ema (float): EMA factor for feature smoothing (alpha = weight on new feature).
        """
        self.bins_h = int(bins_h)
        self.bins_s = int(bins_s)
        self.ema = float(ema)

        # Per-track state
        self.track_feat = {}    # tid -> feature vector (np.float32)
        self.track_stats = {}   # tid -> {"s": mean_s, "v": mean_v}

        # Previous cluster centers (order: [TEAM1, TEAM2, REF])
        self.prev_centers = None

        # Assigned roles per tid
        self.roles = {}         # tid -> {"role": str}

        # Rendering colors in BGR (TEAM1 red, TEAM2 blue by default)
        self.role_colors = {
            "TEAM1": (0, 0, 255),
            "TEAM2": (255, 0, 0),
            "REF":   (255, 255, 255),
            "BALL":  (0, 255, 255),
            "UNK":   (0, 255, 0),
        }

    def update_with_frame(self, frame_bgr, xyxy, ids, cls, id_to_name, jersey_top_frac):
        """
        Update per-track color features from the current frame using EMA.
        The ball is skipped.

        Args:
            frame_bgr (np.ndarray): Current frame (BGR).
            xyxy (np.ndarray): Array of boxes [N, 4] with (x1,y1,x2,y2).
            ids (np.ndarray | None): Track ids [N] or None.
            cls (np.ndarray | None): Class ids [N] or None.
            id_to_name (dict | list | None): Class id to name mapping.
            jersey_top_frac (float): Fraction of box height used for jersey patch.

        Returns:
            None

        Behavior:
            - For each valid track id (tid >= 0) that is not a ball, extract a jersey patch.
            - Compute HSV histogram feature with grass masking.
            - Update the EMA-smoothed feature vector per track.
            - Store per-track mean S and V for later heuristics (referee tie-break).
        """
        if xyxy is None:
            return

        n = len(xyxy)
        for i in range(n):
            # Validate presence of an ID
            if ids is None:
                continue
            if ids[i] is None:
                continue

            # Convert id to int and ensure it is non-negative
            try:
                tid = int(ids[i])
            except Exception:
                continue
            if tid < 0:
                continue

            # Resolve class name if possible
            cname = ""
            if cls is not None and id_to_name:
                try:
                    cname = id_to_name.get(int(cls[i]), str(int(cls[i])))
                except Exception:
                    cname = ""

            # Skip ball entries
            if cname == "ball":
                continue

            # Extract jersey patch
            box = np.asarray(xyxy[i]).astype(int)
            patch = jersey_patch(frame_bgr, box, jersey_top_frac)
            if patch is None:
                continue

            # Compute feature and update EMA
            feat, mean_s, mean_v = hsv_hist_feature(patch, self.bins_h, self.bins_s)
            if tid in self.track_feat:
                prev = self.track_feat[tid]
                self.track_feat[tid] = self.ema * feat + (1.0 - self.ema) * prev
            else:
                self.track_feat[tid] = feat

            # Save S/V statistics
            self.track_stats[tid] = {"s": mean_s, "v": mean_v}

    def cluster_and_assign(self, active_tids, tid_to_clsname):
        """
        Run K-Means with K=3 on available jersey features and assign roles.

        Args:
            active_tids (list[int]): Track ids that are visible in the current frame.
            tid_to_clsname (dict[int, str]): Map of tid -> class name.

        Returns:
            None

        Behavior:
            - Collect candidates among active tracks that have features and are not balls.
            - Run K-Means (K=3) to form three clusters.
            - Choose REF as the smallest cluster (tie-break: lower S/V score).
            - Map the remaining two clusters to TEAM1/TEAM2 by comparing centers
              to the previous centers (2x2 distance assignment).
            - Assign roles TEAM1/TEAM2/REF to all candidate tids.
            - Store current centers as prev_centers (no extra smoothing).
        """
        # Build candidate list: active tids with a feature, and not a ball
        person_tids = []
        for tid in active_tids:
            has_feat = (tid in self.track_feat)
            is_ball = (tid_to_clsname.get(tid, "") == "ball")
            if has_feat and not is_ball:
                person_tids.append(tid)

        # Need at least 3 entries for K=3
        if len(person_tids) < 3:
            return

        # Stack features for K-Means
        data_list = []
        for tid in person_tids:
            data_list.append(self.track_feat[tid])
        data = np.stack(data_list).astype(np.float32)

        # K-Means parameters
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)

        # Run K-Means (with kmeans++ init, 5 attempts)
        _compact, labels, centers = cv2.kmeans(
            data, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()

        # Group members by cluster id
        cluster_members = {0: [], 1: [], 2: []}
        for idx, lab in enumerate(labels):
            k = int(lab)
            tid_k = person_tids[idx]
            cluster_members[k].append(tid_k)

        # Helper: average stat (S or V) over tids in a cluster
        def cluster_mean_stat(members, key):
            values = []
            for tid in members:
                if tid in self.track_stats:
                    values.append(self.track_stats[tid][key])
            if len(values) == 0:
                return 0.0
            return float(np.mean(values))

        # Compute sizes and SV-scores
        sizes = {}
        sv_score = {}
        for k in range(K):
            members = cluster_members[k]
            sizes[k] = len(members)
            mean_v = cluster_mean_stat(members, "v")
            mean_s = cluster_mean_stat(members, "s")
            sv_score[k] = 0.6 * mean_v + 0.4 * mean_s

        # REF = smallest cluster (tie-break by lower SV score)
        def ref_key_func(k):
            return (sizes[k], sv_score[k])

        ref_k = min([0, 1, 2], key=ref_key_func)

        # The two team clusters are the remaining ones
        team_ks = []
        for k in [0, 1, 2]:
            if k != ref_k:
                team_ks.append(k)

        # Map the two team clusters to TEAM1/TEAM2 using previous centers (permutation protection)
        if self.prev_centers is not None:
            prev_t1 = self.prev_centers[0]
            prev_t2 = self.prev_centers[1]

            # Compute pairwise distances current centers -> (prev_t1, prev_t2)
            pair = np.stack([prev_t1, prev_t2])[None, :, :]  # shape (1, 2, D)
            curr = centers[team_ks][:, None, :]               # shape (2, 1, D)
            dists = np.linalg.norm(curr - pair, axis=2)       # shape (2, 2)

            # Two assignments: straight (0->t1, 1->t2) vs swapped (0->t2, 1->t1)
            straight_cost = dists[0, 0] + dists[1, 1]
            swapped_cost = dists[0, 1] + dists[1, 0]

            if straight_cost <= swapped_cost:
                t1_k = team_ks[0]
                t2_k = team_ks[1]
            else:
                t1_k = team_ks[1]
                t2_k = team_ks[0]
        else:
            # No previous centers, keep order as-is
            t1_k = team_ks[0]
            t2_k = team_ks[1]

        # Store current centers (TEAM1, TEAM2, REF) without extra smoothing
        self.prev_centers = [centers[t1_k], centers[t2_k], centers[ref_k]]

        # Assign roles to all members in each cluster
        for k in [0, 1, 2]:
            if k == ref_k:
                role = "REF"
            elif k == t1_k:
                role = "TEAM1"
            elif k == t2_k:
                role = "TEAM2"
            else:
                role = "UNK"

            members = cluster_members[k]
            for tid in members:
                self.roles[tid] = {"role": role}

    def assign_balls_and_unknowns(self, active_tids, tid_to_clsname):
        """
        Finalize role assignments for ball and remaining unknowns.

        Args:
            active_tids (list[int]): Track ids visible in the current frame.
            tid_to_clsname (dict[int, str]): Map of tid -> class name.

        Returns:
            None

        Behavior:
            - If class name is "ball", set role to BALL.
            - Any active tid without a role yet is marked as UNK.
            - No global reset of roles; roles persist until overwritten.
        """
        for tid in active_tids:
            cname = tid_to_clsname.get(tid, "")
            if cname == "ball":
                self.roles[tid] = {"role": "BALL"}
            elif tid not in self.roles:
                self.roles[tid] = {"role": "UNK"}

    def color_for_tid(self, tid):
        """
        Get the rendering color (BGR) and the role string for a given track id.

        Args:
            tid (int): Track id.

        Returns:
            tuple[tuple[int, int, int], str]:
                - color (tuple[int,int,int]): BGR color for the tid's role.
                - role (str): One of {"TEAM1","TEAM2","REF","BALL","UNK"}.

        Behavior:
            - If a role is unknown, returns UNK color and role.
        """
        role = "UNK"
        if tid in self.roles:
            role = self.roles[tid].get("role", "UNK")

        if role in self.role_colors:
            color = self.role_colors[role]
        else:
            color = (0, 255, 0)  # default green for safety

        return color, role