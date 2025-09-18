from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def adapt_conv1_to_in_ch(resnet: nn.Module, in_ch: int) -> nn.Module:
    """Adapt the first conv layer of a torchvision ResNet to arbitrary input channels."""
    if in_ch == 3:
        return resnet
    w = resnet.conv1.weight
    new = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        reps = (in_ch + 2) // 3
        w_rep = w.repeat(1, reps, 1, 1)[:, :in_ch] * (3.0 / in_ch)
        new.weight.copy_(w_rep)
    resnet.conv1 = new
    return resnet


def build_resnet50_feature_extractor(
    in_channels: int = 3,
    pretrained_type: str = "imagenet",
    pretrained_path: Optional[str] = None,
    freeze_until: str = "layer3",
    ) -> Tuple[nn.Module, int]:
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained_type == "imagenet" else None)
    m = adapt_conv1_to_in_ch(m, in_channels)

    if pretrained_type == "radimagenet" and pretrained_path:
        p = Path(pretrained_path)
        if p.exists():
            sd = torch.load(str(p), map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print(f"[RadImageNet] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print(f"[RadImageNet] Checkpoint not found at {p}. Falling back to ImageNet weights.")

    if freeze_until.lower() != "none":
        freeze = True
        for name, p in m.named_parameters():
            if freeze and name.startswith(freeze_until):
                freeze = False
            p.requires_grad = not freeze

    feat = nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2, m.layer3, m.layer4,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    return feat, 2048


class MeanSlicePool(nn.Module):
    def forward(self, f: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """f: (B, S, D); mask: (B, S) in {0,1} or None."""
        if mask is None:
            return f.mean(dim=1)
        w = mask.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        return (f * w.unsqueeze(-1)).sum(dim=1)


class AttnSlicePool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(True), nn.Linear(256, 1)
        )

    def forward(self, f: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attn(f)
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            scores = scores + (mask_float - 1) * 1e9
        w = torch.softmax(scores, dim=1)
        return (f * w).sum(dim=1)
