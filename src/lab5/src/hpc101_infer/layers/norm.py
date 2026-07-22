from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps
        output = hidden_states.float() * torch.pow(mean_squared, -0.5)
        if self.weight is not None:
            output = output * self.weight.float()
        return output.to(hidden_states.dtype)
