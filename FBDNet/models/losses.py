"""Losses and morphology helpers for FBDNet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _foreground_logit(seg_logits: torch.Tensor) -> torch.Tensor:
    """Return a binary foreground logit from 1-channel or 2-channel logits."""
    if seg_logits.shape[1] == 1:
        return seg_logits[:, 0]
    return seg_logits[:, 1] - seg_logits[:, 0]


def soft_dice_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss for binary maps."""
    pred = pred_prob.contiguous().view(-1)
    tgt = target.contiguous().view(-1).float()
    inter = (pred * tgt).sum()
    return 1.0 - (2.0 * inter + eps) / (pred.sum() + tgt.sum() + eps)


def make_boundary_mask(
    seg_label: torch.Tensor,
    kernel_size: int = 3,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Extract a boundary band from a semantic label by local max/min pooling.

    Args:
        seg_label: ``(B, H, W)`` integer semantic target.
        kernel_size: Odd window size controlling boundary thickness.
        ignore_index: Label value ignored by losses.

    Returns:
        ``(B, 1, H, W)`` float mask with boundary pixels equal to 1.
    """
    valid = seg_label != ignore_index
    clean = seg_label.clone().float()
    clean[~valid] = 0.0
    clean = clean.unsqueeze(1)

    pad = kernel_size // 2
    max_val = F.max_pool2d(clean, kernel_size, stride=1, padding=pad)
    min_val = -F.max_pool2d(-clean, kernel_size, stride=1, padding=pad)
    boundary = (max_val != min_val).float()
    return boundary * valid.unsqueeze(1).float()


def make_skeleton_target(
    seg_label: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Approximate a one-pixel foreground skeleton/boundary target.

    A binary foreground erosion is subtracted from the foreground mask.
    """
    valid = seg_label != ignore_index
    clean = seg_label.clone().float()
    clean[~valid] = 0.0
    fg = (clean > 0).float().unsqueeze(1)

    kernel = torch.ones(1, 1, 3, 3, device=fg.device, dtype=fg.dtype)
    eroded = F.conv2d(fg, kernel, padding=1) == 9.0
    skeleton = fg - eroded.float()
    return skeleton.clamp_(0, 1)


class BoundaryWeightedBinaryLoss(nn.Module):
    """Boundary-Weighted Binary Loss (BWBL).

    Skeleton pixels receive a larger loss weight, pixels in the buffer region
    around the skeleton are ignored, and all remaining pixels keep weight 1.
    """

    def __init__(
        self,
        skeleton_weight: float = 5.0,
        buffer_radius: int = 3,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.skeleton_weight = float(skeleton_weight)
        self.buffer_radius = int(buffer_radius)
        self.loss_weight = float(loss_weight)
        size = 2 * self.buffer_radius + 1
        self.register_buffer("buffer_kernel", torch.ones(1, 1, size, size))

    def forward(self, pred_logits: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        if skeleton.ndim == 3:
            skeleton = skeleton.unsqueeze(1)
        skeleton = skeleton.float()
        skel_mask = skeleton > 0.5
        buffer_mask = F.conv2d(
            skel_mask.float(),
            self.buffer_kernel.to(pred_logits.device, pred_logits.dtype),
            padding=self.buffer_radius,
        ) > 0
        buffer_only = buffer_mask & ~skel_mask

        weight = torch.ones_like(skeleton)
        weight[buffer_only] = 0.0
        weight[skel_mask] = self.skeleton_weight

        bce = F.binary_cross_entropy_with_logits(
            pred_logits, skeleton, reduction="none"
        )
        loss = (bce * weight).sum() / weight.sum().clamp(min=1.0)
        return self.loss_weight * loss


def semantic_cross_entropy(
    seg_logits: torch.Tensor,
    seg_label: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Semantic segmentation CE loss."""
    if seg_logits.shape[-2:] != seg_label.shape[-2:]:
        seg_logits = F.interpolate(
            seg_logits,
            size=seg_label.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return F.cross_entropy(seg_logits, seg_label.long(), ignore_index=ignore_index)


def boundary_band_loss(
    seg_logits: torch.Tensor,
    seg_label: torch.Tensor,
    band_width: int = 3,
    loss_weight: float = 0.10,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """BCE + Dice on semantic logits restricted to a GT boundary band."""
    band = make_boundary_mask(seg_label, band_width, ignore_index).squeeze(1).bool()
    valid = seg_label != ignore_index
    selected = band & valid
    if selected.sum() == 0:
        zero = seg_logits.new_zeros(())
        return {"loss_boundary_bce": zero, "loss_boundary_dice": zero}

    logits_up = F.interpolate(
        seg_logits,
        size=seg_label.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    fg_logit = _foreground_logit(logits_up)
    target = (seg_label[selected] > 0).float()
    pred = fg_logit[selected]

    loss_bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    loss_dice = soft_dice_loss(torch.sigmoid(pred), target, eps)
    return {
        "loss_boundary_bce": loss_weight * loss_bce,
        "loss_boundary_dice": loss_weight * loss_dice,
    }


def center_weighted_distance_loss(
    dist_logits: torch.Tensor,
    dist_target: torch.Tensor,
    seg_label: torch.Tensor,
    alpha: float = 3.0,
    loss_weight: float = 0.20,
    smooth_l1_beta: float = 0.1,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    """Foreground SmoothL1 distance loss with center-peak weighting."""
    if dist_target.ndim == 3:
        dist_target = dist_target.unsqueeze(1)
    dist_target = dist_target.float()
    dist_pred = F.interpolate(
        dist_logits,
        size=dist_target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).sigmoid()

    fg_mask = (seg_label != ignore_index) & (seg_label > 0)
    if fg_mask.sum() == 0:
        return {
            "loss_distance": dist_logits.new_zeros(()),
            "distance_fg_ratio": dist_logits.new_tensor(0.0),
        }

    pred_fg = dist_pred[:, 0][fg_mask]
    tgt_fg = dist_target[:, 0][fg_mask]
    weight = 1.0 + alpha * tgt_fg
    per_pixel = F.smooth_l1_loss(
        pred_fg,
        tgt_fg,
        beta=smooth_l1_beta,
        reduction="none",
    )
    return {
        "loss_distance": loss_weight * (weight * per_pixel).mean(),
        "distance_fg_ratio": dist_logits.new_tensor(
            float(fg_mask.sum().item()) / max(float(fg_mask.numel()), 1.0)
        ),
    }
