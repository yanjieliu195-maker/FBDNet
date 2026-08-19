"""Adaptive instance decoding for FBDNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass
class DecodingConfig:
    semantic_thr: float = 0.5
    boundary_thr: float = 0.30
    boundary_dilate: int = 1
    open_radius: int = 5
    min_island_area: int = 2000
    max_hole_area: int = 20000
    sigma: float = 1.5
    peak_thr: float = 0.30
    min_dist: int = 15
    max_markers: int = 50
    min_area_multi: int = 50000
    min_sep: float = 0.20


def _as_uint8_mask(prob_or_mask: np.ndarray, thr: float = 0.5) -> np.ndarray:
    arr = np.asarray(prob_or_mask)
    if arr.dtype == np.uint8 and arr.max() > 1:
        return ((arr > int(thr * 255)).astype(np.uint8) * 255)
    return ((arr > thr).astype(np.uint8) * 255)


def semantic_clean(
    sem_uint8: np.ndarray,
    open_radius: int = 5,
    min_island_area: int = 2000,
    max_hole_area: int = 20000,
) -> np.ndarray:
    """Remove narrow bridges, tiny islands, and small holes."""
    fg = (sem_uint8 > 127).astype(np.uint8) * 255
    if open_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1)
        )
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    if min_island_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for idx in range(1, n):
            if stats[idx, cv2.CC_STAT_AREA] < min_island_area:
                fg[labels == idx] = 0

    if max_hole_area > 0:
        inv = ((fg == 0).astype(np.uint8) * 255)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        h, w = fg.shape
        for idx in range(1, n):
            x, y, cw, ch, area = stats[idx]
            touches_border = x == 0 or y == 0 or x + cw >= w or y + ch >= h
            if not touches_border and area < max_hole_area:
                fg[labels == idx] = 255
    return fg


def boundary_cut_semantic(
    sem_uint8: np.ndarray,
    boundary_prob: Optional[np.ndarray],
    boundary_thr: float = 0.30,
    dilate_px: int = 1,
) -> np.ndarray:
    """Force high-confidence boundary pixels to background before watershed."""
    if boundary_prob is None:
        return sem_uint8
    boundary_u8 = _as_uint8_mask(boundary_prob, boundary_thr)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        boundary_u8 = cv2.dilate(boundary_u8, np.ones((k, k), np.uint8))
    out = sem_uint8.copy()
    out[boundary_u8 > 0] = 0
    return out


def _semantic_regions(sem_uint8: np.ndarray):
    fg = (sem_uint8 > 127).astype(np.uint8)
    return cv2.connectedComponentsWithStats(fg, connectivity=8)


def _merge_shallow_watershed(
    ws: np.ndarray,
    dist_s: np.ndarray,
    fg: np.ndarray,
    min_sep: float,
) -> np.ndarray:
    max_label = int(ws.max())
    if max_label <= 1 or min_sep <= 0:
        return ws

    peak = np.zeros(max_label + 1, dtype=np.float32)
    for lab in range(1, max_label + 1):
        mask = (ws == lab) & fg
        if mask.any():
            peak[lab] = float(dist_s[mask].max())

    saddle = {}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(ws, (dr, dc), axis=(0, 1))
        valid = np.ones(ws.shape, dtype=bool)
        if dr == -1:
            valid[-1, :] = False
        elif dr == 1:
            valid[0, :] = False
        if dc == -1:
            valid[:, -1] = False
        elif dc == 1:
            valid[:, 0] = False
        edge = fg & valid & (ws > 0) & (shifted > 0) & (ws != shifted)
        for a, b, d in zip(ws[edge], shifted[edge], dist_s[edge]):
            a, b = int(min(a, b)), int(max(a, b))
            saddle[(a, b)] = max(float(d), saddle.get((a, b), 0.0))

    parent = list(range(max_label + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for (a, b), s in saddle.items():
        if peak[a] > 0 and peak[b] > 0 and min(peak[a], peak[b]) - s < min_sep:
            union(a, b)

    out = np.zeros_like(ws)
    remap = {}
    next_id = 1
    for lab in range(1, max_label + 1):
        root = find(lab)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        out[ws == lab] = remap[root]
    return out


def distance_constrained_watershed(
    dist: np.ndarray,
    sem_uint8: np.ndarray,
    cfg: DecodingConfig,
) -> Tuple[np.ndarray, int]:
    """Run per-region normalized peak watershed constrained by semantics."""
    fg = sem_uint8 > 127
    n_regions, sem_labels, stats, _ = _semantic_regions(sem_uint8)
    dist = np.asarray(dist, dtype=np.float32)
    dist_norm = np.zeros_like(dist, dtype=np.float32)

    peaks = np.zeros(dist.shape, dtype=bool)
    for rid in range(1, n_regions):
        region = sem_labels == rid
        values = dist[region]
        if values.size == 0:
            continue
        lo, hi = float(values.min()), float(values.max())
        if hi - lo > 1e-6:
            dist_norm[region] = (dist[region] - lo) / (hi - lo)
        else:
            dist_norm[region] = 1.0

        area = int(stats[rid, cv2.CC_STAT_AREA])
        max_peaks = 1 if area < cfg.min_area_multi else min(
            cfg.max_markers, max(1, round(area / cfg.min_area_multi))
        )
        coords = peak_local_max(
            gaussian_filter(dist_norm * region, sigma=cfg.sigma),
            min_distance=cfg.min_dist,
            threshold_abs=cfg.peak_thr,
            num_peaks=max_peaks,
            labels=region.astype(np.uint8),
        )
        if len(coords) == 0:
            coords = np.argwhere(region)
            if len(coords) > 0:
                best = np.argmax(dist_norm[region])
                coords = coords[[best]]
        peaks[coords[:, 0], coords[:, 1]] = True

    markers, n_markers = label(peaks)
    if n_markers == 0:
        out = np.zeros_like(sem_uint8, dtype=np.int32)
        if fg.any():
            out[fg] = 1
        return out, int(out.max())

    dist_s = gaussian_filter(dist_norm, sigma=cfg.sigma)
    ws = watershed(-dist_s, markers.astype(np.int32), mask=fg)
    ws = _merge_shallow_watershed(ws.astype(np.int32), dist_s, fg, cfg.min_sep)
    return ws.astype(np.int32), int(ws.max())


def adaptive_instance_decoding(
    semantic_prob: np.ndarray,
    distance_prob: np.ndarray,
    boundary_prob: Optional[np.ndarray] = None,
    config: Optional[DecodingConfig] = None,
) -> np.ndarray:
    """Convert FBDNet triplet outputs to an instance map.

    Args:
        semantic_prob: Foreground probability or uint8 semantic mask.
        distance_prob: Normalized distance map in ``[0, 1]``.
        boundary_prob: Optional boundary probability map.
        config: Decoding hyperparameters.

    Returns:
        ``int32`` instance map with 0 as background.
    """
    cfg = config or DecodingConfig()
    sem = _as_uint8_mask(semantic_prob, cfg.semantic_thr)
    sem = boundary_cut_semantic(sem, boundary_prob, cfg.boundary_thr, cfg.boundary_dilate)
    sem = semantic_clean(sem, cfg.open_radius, cfg.min_island_area, cfg.max_hole_area)
    inst, _ = distance_constrained_watershed(distance_prob, sem, cfg)
    return inst
