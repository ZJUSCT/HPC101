from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str | None = None
    input_ids: list[int] | None = None
    max_new_tokens: int = 1
    stop_token_ids: tuple[int, ...] | None = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    def __post_init__(self) -> None:
        if (self.prompt is None) == (self.input_ids is None):
            raise ValueError("exactly one of prompt and input_ids must be provided")
        if self.input_ids is not None and not self.input_ids:
            raise ValueError("input_ids must not be empty")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")


@dataclass(frozen=True)
class RequestMetrics:
    prefill_latency_s: float
    decode_latencies_s: tuple[float, ...] = ()
    total_latency_s: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    @property
    def ttft_s(self) -> float:
        return self.prefill_latency_s

    @property
    def mean_tpot_s(self) -> float:
        if not self.decode_latencies_s:
            return 0.0
        return sum(self.decode_latencies_s) / len(self.decode_latencies_s)

    def pretty(self) -> str:
        return f"""
TTFT (s): {self.ttft_s:.4f}
Mean TPOT (s): {self.mean_tpot_s:.4f}
Total latency (s): {self.total_latency_s:.4f}
Peak allocated bytes: {self.peak_allocated_bytes/1048576:.2f} MB
Peak reserved bytes: {self.peak_reserved_bytes/1048576:.2f} MB
        """


@dataclass(frozen=True)
class PrefillOutput:
    logits: torch.Tensor
    sequence_lengths: torch.Tensor
    latency_s: float
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


@dataclass(frozen=True)
class DecodeOutput:
    logits: torch.Tensor
    sequence_lengths: torch.Tensor
    latency_s: float
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


@dataclass(frozen=True)
class GenerationOutput:
    token_ids: list[int]
    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    metrics: RequestMetrics


@dataclass(frozen=True)
class ScoreOutput:
    logits: torch.Tensor | None
    token_logprobs: torch.Tensor | None
    nll_sum: float
    token_count: int
    mean_nll: float
    mean_logprob: float
    perplexity: float
