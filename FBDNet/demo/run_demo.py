#!/usr/bin/env python
"""Synthetic decoding demo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decoding.adaptive_instance_decoding import (
    DecodingConfig,
    adaptive_instance_decoding,
)


def make_synthetic_triplet(size: int = 256):
    y, x = np.mgrid[:size, :size]
    centers = [(80, 80), (170, 95), (135, 180)]
    sem = np.zeros((size, size), dtype=np.float32)
    dist = np.zeros((size, size), dtype=np.float32)
    boundary = np.zeros((size, size), dtype=np.float32)

    for cy, cx in centers:
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        mask = r < 45
        sem[mask] = 0.95
        local = np.clip(1.0 - r / 45.0, 0.0, 1.0)
        dist = np.maximum(dist, local)
        boundary[(r >= 42) & (r <= 46)] = 0.9
    return sem, dist, boundary


def main() -> None:
    sem, dist, boundary = make_synthetic_triplet()
    inst = adaptive_instance_decoding(
        sem,
        dist,
        boundary,
        DecodingConfig(min_island_area=80, max_hole_area=500, min_area_multi=2500),
    )
    print("Synthetic instances:", int(inst.max()))
    print("Instance map shape:", inst.shape)


if __name__ == "__main__":
    main()
