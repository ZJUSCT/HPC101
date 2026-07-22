from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, TypeVar
from contextlib import contextmanager

import torch

from hpc101_infer.runtime.device import synchronize

T = TypeVar("T")


@dataclass
class OperationMetrics:
    latency_s: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


@contextmanager
def measure_operation(device: torch.device, synchronize_cuda: bool):
    synchronize(device, synchronize_cuda)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    metrics = OperationMetrics()
    try:
        yield metrics
    finally:
        metrics.latency_s = perf_counter() - start
        if device.type == "cuda":
            metrics.peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            metrics.peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        else:
            metrics.peak_allocated_bytes = 0
            metrics.peak_reserved_bytes = 0
        return metrics
