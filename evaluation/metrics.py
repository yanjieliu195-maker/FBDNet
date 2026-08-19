"""Metrics for FBDNet."""

from __future__ import annotations

import cv2
import numpy as np


def instance_iou_matrix(pred_map: np.ndarray, gt_map: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between predicted and GT instances."""
    pred_ids = np.unique(pred_map)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_map)
    gt_ids = gt_ids[gt_ids > 0]
    mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return mat

    for i, pid in enumerate(pred_ids):
        p = pred_map == pid
        for j, gid in enumerate(gt_ids):
            g = gt_map == gid
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            mat[i, j] = inter / max(float(union), 1.0)
    return mat


def f1_at_iou(iou_mat: np.ndarray, threshold: float = 0.5) -> dict:
    """Greedy instance matching at one IoU threshold."""
    if iou_mat.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    matched_gt = set()
    tp = 0
    order = np.argsort(-iou_mat.max(axis=1))
    for pred_idx in order:
        gt_idx = int(np.argmax(iou_mat[pred_idx]))
        if iou_mat[pred_idx, gt_idx] >= threshold and gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(gt_idx)
    fp = iou_mat.shape[0] - tp
    fn = iou_mat.shape[1] - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _boundary_from_label(label_map: np.ndarray, kernel: int = 3) -> np.ndarray:
    fg = (label_map > 0).astype(np.uint8)
    k = np.ones((kernel, kernel), np.uint8)
    return fg - cv2.erode(fg, k, iterations=1)


def compute_semantic_metrics(pred_sem: np.ndarray, gt_sem: np.ndarray) -> dict:
    """Binary semantic metrics plus relaxed boundary metrics."""
    pred = (pred_sem > 127).astype(np.uint8)
    gt = (gt_sem > 0).astype(np.uint8)

    tp = int((pred & gt).sum())
    fp = int((pred & (1 - gt)).sum())
    fn = int(((1 - pred) & gt).sum())
    tn = int(((1 - pred) & (1 - gt)).sum())
    total = max(int(pred.size), 1)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou_fg = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    iou_bg = tn / (tn + fp + fn) if tn + fp + fn > 0 else 0.0

    bpred = _boundary_from_label(pred * 255)
    bgt = _boundary_from_label(gt * 255)
    relax = np.ones((7, 7), np.uint8)
    bpred_d = cv2.dilate(bpred, relax)
    bgt_d = cv2.dilate(bgt, relax)
    b_tp = int(((bpred_d > 0) & (bgt > 0)).sum())
    b_fp = int(((bpred > 0) & ~(bgt_d > 0)).sum())
    b_fn = int(((bgt > 0) & ~(bpred_d > 0)).sum())
    biou = b_tp / (b_tp + b_fp + b_fn) if b_tp + b_fp + b_fn > 0 else 0.0
    bprecision = b_tp / (b_tp + b_fp) if b_tp + b_fp > 0 else 0.0
    brecall = b_tp / (b_tp + b_fn) if b_tp + b_fn > 0 else 0.0
    bf1 = 2 * bprecision * brecall / (bprecision + brecall) if bprecision + brecall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_fg": iou_fg,
        "iou_bg": iou_bg,
        "miou": 0.5 * (iou_fg + iou_bg),
        "overall_accuracy": (tp + tn) / total,
        "boundary_iou": biou,
        "boundary_f1": bf1,
    }
