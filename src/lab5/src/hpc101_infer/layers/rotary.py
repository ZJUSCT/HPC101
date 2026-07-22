from __future__ import annotations

import torch

from hpc101_infer.models.config import RotaryConfig


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        config: RotaryConfig,
        head_dim: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        cache = self._build_cache(max_position_embeddings, torch.device("cpu"))
        self.register_buffer("_cos_sin_cache", cache, persistent=False)

    def _build_cache(
        self, max_position_embeddings: int, device: torch.device
    ) -> torch.Tensor:
        config = self.config
        head_dim = self.head_dim

        base = config.rope_theta
        if config.rope_type == "proportional":
            rotary_pairs = int(
                float(config.partial_rotary_factor or 1.0) * head_dim // 2
            )
            rotated = 1.0 / (
                base
                ** (
                    torch.arange(
                        0,
                        2 * rotary_pairs,
                        2,
                        dtype=torch.float32,
                        device=device,
                    )
                    / head_dim
                )
            )
            inv_freq = torch.nn.functional.pad(
                rotated, (0, head_dim // 2 - rotary_pairs)
            ) / float(config.factor)
        else:
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                    / head_dim
                )
            )

        positions = torch.arange(
            max_position_embeddings, dtype=torch.float32, device=device
        )
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        embedding = torch.cat((freqs, freqs), dim=-1)
        return torch.cat((embedding.cos(), embedding.sin()), dim=-1)

    def materialize(self, device: str | torch.device) -> None:
        device = torch.device(device)
        self._cos_sin_cache = self._build_cache(self.max_position_embeddings, device)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(
        self, position_ids: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self._cos_sin_cache[position_ids]
        cos = cos_sin[..., : self.head_dim].to(query.dtype)
        sin = cos_sin[..., self.head_dim :].to(query.dtype)
        return self._apply_rotary(query, cos, sin), self._apply_rotary(key, cos, sin)
