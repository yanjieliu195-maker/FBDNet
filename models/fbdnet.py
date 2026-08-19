"""FBDNet model definition."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bgfr import BoundaryGuidedFeatureResidual
from .losses import (
    BoundaryWeightedBinaryLoss,
    boundary_band_loss,
    center_weighted_distance_loss,
    make_skeleton_target,
    semantic_cross_entropy,
)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPooling(nn.Module):
    """Small PPM block used by the UPer-style decoder."""

    def __init__(self, in_channels: int, out_channels: int, scales=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvBNReLU(in_channels, out_channels, kernel_size=1),
                )
                for scale in scales
            ]
        )
        self.bottleneck = ConvBNReLU(
            in_channels + len(scales) * out_channels,
            out_channels,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [
            F.interpolate(branch(x), size=size, mode="bilinear", align_corners=False)
            for branch in self.branches
        ]
        return self.bottleneck(torch.cat([x, *pooled], dim=1))


class UPerStyleFusion(nn.Module):
    """Minimal FPN+PPM feature fusion used to show the FBDNet decode logic."""

    def __init__(
        self,
        in_channels: Iterable[int],
        channels: int = 256,
        ppm_scales=(1, 2, 3, 6),
    ) -> None:
        super().__init__()
        in_channels = list(in_channels)
        self.lateral_convs = nn.ModuleList(
            [ConvBNReLU(ch, channels, kernel_size=1) for ch in in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvBNReLU(channels, channels, kernel_size=3) for _ in in_channels[:-1]]
        )
        self.ppm = PyramidPooling(in_channels[-1], channels, ppm_scales)
        self.fpn_bottleneck = ConvBNReLU(len(in_channels) * channels, channels, 3)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feats[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self.ppm(feats[-1]))

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outs = [conv(laterals[i]) for i, conv in enumerate(self.fpn_convs)]
        outs.append(laterals[-1])
        out_size = outs[0].shape[-2:]
        outs = [
            F.interpolate(o, size=out_size, mode="bilinear", align_corners=False)
            if o.shape[-2:] != out_size
            else o
            for o in outs
        ]
        return self.fpn_bottleneck(torch.cat(outs, dim=1))


class FBDNetHead(nn.Module):
    """M5-style head: semantic + boundary + distance."""

    def __init__(
        self,
        in_channels: Iterable[int],
        channels: int = 256,
        num_classes: int = 2,
        edge_in_index: int = 0,
        edge_mid_channels: int = 128,
        dist_mid_channels: int = 64,
        band_width: int = 3,
        boundary_loss_weight: float = 0.10,
        edge_loss_weight: float = 0.30,
        distance_loss_weight: float = 0.20,
        distance_alpha: float = 3.0,
        smooth_l1_beta: float = 0.1,
        skeleton_weight: float = 5.0,
        buffer_radius: int = 3,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        in_channels = list(in_channels)
        self.edge_in_index = edge_in_index
        self.band_width = band_width
        self.boundary_loss_weight = boundary_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.distance_alpha = distance_alpha
        self.smooth_l1_beta = smooth_l1_beta
        self.ignore_index = ignore_index

        self.fusion = UPerStyleFusion(in_channels, channels)
        self.classifier = nn.Conv2d(channels, num_classes, 1)
        self.bgfr = BoundaryGuidedFeatureResidual(
            low_level_channels=in_channels[edge_in_index],
            decoder_channels=channels,
            num_classes=num_classes,
            edge_mid_channels=edge_mid_channels,
            residual_weight=1.0,
        )
        self.distance_head = nn.Sequential(
            nn.Conv2d(channels, dist_mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(dist_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dist_mid_channels, 1, 1),
        )
        nn.init.constant_(self.distance_head[-1].weight, 0.0)
        nn.init.constant_(self.distance_head[-1].bias, 0.0)
        self.bwbl = BoundaryWeightedBinaryLoss(
            skeleton_weight=skeleton_weight,
            buffer_radius=buffer_radius,
        )

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        decoder_feat = self.fusion(feats)
        base_logits = self.classifier(decoder_feat)
        seg_logits, boundary_logits = self.bgfr(
            feats[self.edge_in_index], decoder_feat, base_logits
        )
        distance_logits = self.distance_head(decoder_feat)
        return {
            "semantic": seg_logits,
            "boundary": boundary_logits,
            "distance": distance_logits,
        }

    def loss(
        self,
        outputs: Dict[str, torch.Tensor],
        seg_label: torch.Tensor,
        dist_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        losses["loss_semantic"] = semantic_cross_entropy(
            outputs["semantic"], seg_label, self.ignore_index
        )

        skeleton = make_skeleton_target(seg_label, self.ignore_index)
        boundary_logits = F.interpolate(
            outputs["boundary"],
            size=skeleton.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        losses["loss_edge_bwbl"] = self.edge_loss_weight * self.bwbl(
            boundary_logits, skeleton
        )
        losses.update(
            boundary_band_loss(
                outputs["semantic"],
                seg_label,
                band_width=self.band_width,
                loss_weight=self.boundary_loss_weight,
                ignore_index=self.ignore_index,
            )
        )

        if dist_target is not None:
            losses.update(
                center_weighted_distance_loss(
                    outputs["distance"],
                    dist_target,
                    seg_label,
                    alpha=self.distance_alpha,
                    loss_weight=self.distance_loss_weight,
                    smooth_l1_beta=self.smooth_l1_beta,
                    ignore_index=self.ignore_index,
                )
            )
        return losses


class FBDNet(nn.Module):
    """Thin wrapper around a backbone and the FBDNet head."""

    def __init__(self, backbone: nn.Module, decode_head: FBDNetHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        if isinstance(feats, torch.Tensor):
            raise TypeError("FBDNet expects the backbone to return multi-scale features.")
        return list(feats)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decode_head(self.extract_features(x))
