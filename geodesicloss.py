import torch
from torch import nn


class GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        R_diff = torch.matmul(input.transpose(-2, -1), target)

        trace = R_diff.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)

        cos_angle = torch.clamp((trace - 1) / 2, min=-1 + self.eps, max=1 - self.eps)
        angles = torch.acos(cos_angle)

        if self.reduction == "mean":
            return angles.mean()
        elif self.reduction == "sum":
            return angles.sum()
        elif self.reduction == "none":
            return angles
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")
