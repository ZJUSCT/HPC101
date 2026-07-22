from __future__ import annotations

import torch


def synchronize(device: torch.device, enabled: bool = True) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
