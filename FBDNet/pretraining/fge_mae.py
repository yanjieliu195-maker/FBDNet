"""FGE-MAE components."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborTargetGenerator(nn.Module):
    """Multi-scale, multi-orientation Gabor response target generator."""

    def __init__(
        self,
        scales: Tuple[int, ...] = (3, 5, 7),
        num_orientations: int = 4,
        patch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.normalize = normalize
        max_scale = int(max(scales))
        self.ksize = max_scale * 6 + 1
        if self.ksize % 2 == 0:
            self.ksize += 1

        kernels = []
        for sigma in scales:
            lambd = float(sigma) * 2.0
            for k in range(num_orientations):
                theta = k * math.pi / num_orientations
                kernels.append(
                    torch.from_numpy(
                        self._make_gabor(self.ksize, float(sigma), theta, lambd, 0.5)
                    ).float()
                )
        self.register_buffer("kernels", torch.stack(kernels).unsqueeze(1))

    @staticmethod
    def _make_gabor(
        ksize: int,
        sigma: float,
        theta: float,
        lambd: float,
        gamma: float,
    ) -> np.ndarray:
        half = ksize // 2
        xs = np.arange(-half, half + 1, dtype=np.float32)
        ys = np.arange(-half, half + 1, dtype=np.float32)
        xs, ys = np.meshgrid(xs, ys)
        x_theta = xs * np.cos(theta) + ys * np.sin(theta)
        y_theta = -xs * np.sin(theta) + ys * np.cos(theta)
        kernel = np.exp(
            -0.5 * (x_theta**2 + (gamma * y_theta) ** 2) / (sigma**2)
        ) * np.cos(2.0 * np.pi * x_theta / lambd)
        kernel = kernel - kernel.mean()
        denom = np.abs(kernel).max()
        if denom > 1e-8:
            kernel = kernel / denom
        return kernel.astype(np.float32)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        pad = self.ksize // 2
        responses = F.conv2d(
            F.pad(gray, (pad, pad, pad, pad), mode="reflect"),
            self.kernels.to(x.device, x.dtype),
        ).abs()
        edge = responses.max(dim=1, keepdim=True)[0]
        if self.normalize:
            b = edge.shape[0]
            mn = edge.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
            mx = edge.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
            edge = (edge - mn) / (mx - mn + 1e-8)
        return edge

    def patchify(self, feat: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        b, c, h, w = feat.shape
        if h % p != 0 or w % p != 0:
            raise ValueError("Feature size must be divisible by patch_size.")
        feat = feat.reshape(b, c, h // p, p, w // p, p)
        feat = torch.einsum("bchpwq->bhwpqc", feat)
        return feat.reshape(b, (h // p) * (w // p), p * p * c)


class FGEMAELoss(nn.Module):
    """Dual reconstruction loss used by FGE-MAE.

    Args:
        patch_size: MAE patch size.
        loss_freq_weight: Weight applied to the Gabor reconstruction loss.
        norm_pix: Whether to normalize RGB target patches as in MAE.
    """

    def __init__(
        self,
        patch_size: int = 16,
        loss_freq_weight: float = 0.5,
        norm_pix: bool = True,
        gabor: GaborTargetGenerator | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.loss_freq_weight = loss_freq_weight
        self.norm_pix = norm_pix
        self.gabor = gabor or GaborTargetGenerator(patch_size=patch_size)

    def patchify_rgb(self, imgs: torch.Tensor) -> torch.Tensor:
        patches = self.gabor.patchify(imgs)
        if self.norm_pix:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1e-6).sqrt()
        return patches

    @staticmethod
    def masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = (pred - target).pow(2).mean(dim=-1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    @staticmethod
    def masked_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = (pred - target).abs().mean(dim=-1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    def forward(
        self,
        pred_rgb: torch.Tensor,
        pred_freq: torch.Tensor,
        imgs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_rgb = self.patchify_rgb(imgs)
        freq_map = self.gabor(imgs)
        target_freq = self.gabor.patchify(freq_map)
        loss_rgb = self.masked_mse(pred_rgb, target_rgb, mask)
        loss_freq = self.masked_l1(pred_freq, target_freq, mask)
        return {
            "loss_rgb": loss_rgb,
            "loss_freq": self.loss_freq_weight * loss_freq,
        }
