from typing import Optional
import torch
import torch.nn as nn
from .model_utils import build_resnet50_feature_extractor, MeanSlicePool, AttnSlicePool


class MedicalDualPathNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_slices: Optional[int],
        gin_channels: int,
        lin_channels: int,
        pretrained_type: str = "imagenet",
        pretrained_global_path: Optional[str] = None,
        pretrained_local_path: Optional[str] = None,
        pool: str = "mean",
        dropout: float = 0.3,
        freeze_until: str = "layer4",
    ):
        super().__init__()
        self.num_slices = num_slices

        self.global_backbone, gdim = build_resnet50_feature_extractor(
            in_channels=gin_channels,
            pretrained_type=pretrained_type,
            pretrained_path=pretrained_global_path,
            freeze_until=freeze_until,
        )
        self.local_backbone,  ldim = build_resnet50_feature_extractor(
            in_channels=lin_channels,
            pretrained_type=pretrained_type,
            pretrained_path=pretrained_local_path,
            freeze_until=freeze_until,
        )

        if pool == "mean":
            self.g_pool = MeanSlicePool()
            self.l_pool = MeanSlicePool()
        elif pool == "attn":
            self.g_pool = AttnSlicePool(gdim)
            self.l_pool = AttnSlicePool(ldim)
        else:
            raise ValueError(f"Unsupported pool: {pool}")

        self.g_proj = nn.Sequential(nn.LayerNorm(gdim), nn.Linear(gdim, 512), nn.ReLU(True), nn.Dropout(dropout))
        self.l_proj = nn.Sequential(nn.LayerNorm(ldim), nn.Linear(ldim, 512), nn.ReLU(True), nn.Dropout(dropout))

        self.gate = nn.Parameter(torch.tensor([0.0, 0.0]))

        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(dropout * 0.5),
        )
        self.cls = nn.Linear(256, num_classes)

    def _extract_slice_features(self, x: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
        """x: (B, S, C, H, W) or (B, C, H, W) -> (B, S, D)"""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        f = backbone(x)
        f = f.view(B, S, -1)
        return f

    def forward(self, global_x: torch.Tensor, local_x: torch.Tensor, slice_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        gf = self._extract_slice_features(global_x, self.global_backbone)
        lf = self._extract_slice_features(local_x,  self.local_backbone)

        gf = self.g_pool(gf, slice_mask)
        lf = self.l_pool(lf, slice_mask)

        gf = self.g_proj(gf)
        lf = self.l_proj(lf)
        g_w, l_w = torch.softmax(self.gate, dim=0)
        fused = torch.cat([g_w * gf, l_w * lf], dim=1)

        h = self.head(fused)
        return self.cls(h)

