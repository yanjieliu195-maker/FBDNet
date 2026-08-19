"""Boundary-guided feature residual modules for FBDNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEdgeHead(nn.Module):
    """Lightweight edge branch attached to a shallow backbone feature."""

    def __init__(self, in_channels: int, mid_channels: int = 128) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class BoundaryGuidedFeatureResidual(nn.Module):
    """Edge-gated additive residual in semantic-logit space.

    Earlier boundary guidance variants modulated decoder features directly.
    FBDNet uses a logit residual instead:

        seg_logits = base_logits + residual_weight * R(decoder_feat) * P(edge)

    This keeps interior logits calibrated while allowing boundary-local
    corrections.
    """

    def __init__(
        self,
        low_level_channels: int,
        decoder_channels: int,
        num_classes: int,
        edge_mid_channels: int = 128,
        residual_weight: float = 1.0,
        learnable_residual_weight: bool = False,
    ) -> None:
        super().__init__()
        self.edge_head = SimpleEdgeHead(low_level_channels, edge_mid_channels)
        self.residual_proj = nn.Conv2d(decoder_channels, num_classes, 1)
        nn.init.constant_(self.residual_proj.weight, 0.0)
        nn.init.constant_(self.residual_proj.bias, 0.0)

        if learnable_residual_weight:
            self.residual_weight = nn.Parameter(torch.tensor(float(residual_weight)))
        else:
            self.residual_weight = float(residual_weight)

    def forward(
        self,
        low_level_feat: torch.Tensor,
        decoder_feat: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_logits = self.edge_head(low_level_feat)
        edge_prob = torch.sigmoid(edge_logits)
        edge_prob = F.interpolate(
            edge_prob,
            size=decoder_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = self.residual_proj(decoder_feat)
        seg_logits = base_logits + self.residual_weight * residual * edge_prob
        return seg_logits, edge_logits
