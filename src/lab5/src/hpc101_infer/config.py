from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EngineConfig:
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    max_batch_size: int = 1
    scheduler_batch_size: int = 1
    max_sequence_length: int = 4096
    attention_backend: str = "eager"
    linear_backend: str = "bf16"
    scheduler_backend: str = "static_batch"
    seed: int = 0
    synchronize_metrics: bool = True

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.scheduler_batch_size <= 0:
            raise ValueError("scheduler_batch_size must be positive")
        if self.scheduler_batch_size > self.max_batch_size:
            raise ValueError("scheduler_batch_size must not exceed max_batch_size")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.attention_backend != "eager":
            raise ValueError("only the eager attention backend is implemented")
        if self.linear_backend not in {"bf16", "int4_reference"}:
            raise ValueError("linear_backend must be bf16 or int4_reference")
        if self.scheduler_backend != "static_batch":
            raise ValueError("only the static_batch scheduler is implemented")
