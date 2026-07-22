from dataclasses import dataclass

import torch

from hpc101_infer.types import GenerationRequest


def _make_device_tensor(
    data: list[int] | list[float], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        data,
        dtype=dtype,
        pin_memory=device.type == "cuda",
    ).to(device, non_blocking=device.type == "cuda")


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


class Sampler:
    def __init__(self, device: torch.device, vocab_size: int):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.device = device
        self.vocab_size = vocab_size

    def prepare(self, requests: list[GenerationRequest]) -> BatchSamplingArgs:
        if not requests:
            raise ValueError("requests must not be empty")
        params = [request.sampling_params for request in requests]
        if all(param.is_greedy for param in params):
            return BatchSamplingArgs(temperatures=None)

        temperatures = _make_device_tensor(
            [1.0 if param.is_greedy else param.temperature for param in params],
            torch.float32,
            self.device,
        )
        effective_top_k = [
            (
                1
                if param.is_greedy
                else (
                    min(param.top_k, self.vocab_size)
                    if param.top_k > 0
                    else self.vocab_size
                )
            )
            for param in params
        ]
        effective_top_p = [1.0 if param.is_greedy else param.top_p for param in params]

        top_k = None
        if any(value < self.vocab_size for value in effective_top_k):
            top_k = _make_device_tensor(effective_top_k, torch.int32, self.device)

        top_p = None
        if any(value < 1.0 for value in effective_top_p):
            top_p = _make_device_tensor(effective_top_p, torch.float32, self.device)

        return BatchSamplingArgs(temperatures=temperatures, top_k=top_k, top_p=top_p)

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)

        filtered_logits = logits.float() / args.temperatures.unsqueeze(-1)

        if args.top_k is None and args.top_p is None:
            probabilities = torch.softmax(filtered_logits, dim=-1)
            return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

        sorted_logits, sorted_indices = torch.sort(
            filtered_logits,
            dim=-1,
            descending=True,
        )
        token_ranks = torch.arange(self.vocab_size, device=self.device).unsqueeze(0)

        if args.top_k is not None:
            top_k_mask = token_ranks >= args.top_k.unsqueeze(-1)
            sorted_logits = sorted_logits.masked_fill(top_k_mask, float("-inf"))

        if args.top_p is not None:
            sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
            cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
            top_p_mask = (
                cumulative_probabilities - sorted_probabilities
            ) >= args.top_p.unsqueeze(-1)
            sorted_logits = sorted_logits.masked_fill(top_p_mask, float("-inf"))

        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        sampled_sorted_indices = torch.multinomial(sorted_probabilities, num_samples=1)
        next_tokens = sorted_indices.gather(dim=-1, index=sampled_sorted_indices)
        return next_tokens.squeeze(-1)
