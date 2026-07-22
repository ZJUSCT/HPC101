"""连续预分配 KV cache，用于展示 prefill 与 decode 共享状态的方式。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hpc101_infer.models.config import Gemma4TextConfig


@dataclass
class LayerKVCache:
    """单层 attention 的 K/V 存储。

    ``key`` 和 ``value`` 的布局均为
    ``[max_batch, kv_heads, max_sequence_length, head_dim]``。
    """
    key: torch.Tensor
    value: torch.Tensor
    lengths: torch.Tensor
    max_batch_size: int
    max_sequence_length: int
    batch_size: int = 0

    def reset(self, batch_size: int) -> None:
        """开始新的静态 batch；旧 tensor 不清零，只重置有效长度。"""
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError(
                f"batch_size must be in [1, {self.max_batch_size}], "
                f"got {batch_size}"
            )
        self.batch_size = batch_size
        self.lengths.zero_()

    def write(
        self,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """按绝对 token 位置写入当前层新计算出的 K/V。"""
        batch_size, query_length = positions.shape
        expected_prefix = (batch_size, self.key.shape[1])
        expected_suffix = (query_length, self.key.shape[3])
        if key.shape != expected_prefix + expected_suffix:
            raise ValueError(
                f"invalid key shape {tuple(key.shape)}, expected "
                f"{expected_prefix + expected_suffix}"
            )
        for batch_idx in range(batch_size):
            target = positions[batch_idx]
            self.key[batch_idx].index_copy_(1, target, key[batch_idx])
            self.value[batch_idx].index_copy_(1, target, value[batch_idx])

    def view(self, max_length: int) -> "LayerKVView":
        """只暴露当前 batch 在 ``max_length`` 以内的连续 cache 前缀。"""
        return LayerKVView(
            key=self.key[: self.batch_size, :, :max_length, :],
            value=self.value[: self.batch_size, :, :max_length, :],
        )

    def commit(self, sequence_lengths: torch.Tensor) -> None:
        """在一次模型 forward 完成后提交新的有效序列长度。"""
        self.lengths[: self.batch_size].copy_(sequence_lengths)


@dataclass(frozen=True)
class LayerKVView:
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    """管理所有 decoder layer 的预分配 cache，并统一提交序列长度。"""
    def __init__(
        self,
        layers: list[LayerKVCache],
        max_batch_size: int,
        max_sequence_length: int,
    ) -> None:
        if not layers:
            raise ValueError("KV cache must contain at least one layer")
        self.layers = layers
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

    @classmethod
    def allocate(
        cls,
        config: Gemma4TextConfig,
        max_batch_size: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> "KVCache":
        if max_batch_size <= 0 or max_sequence_length <= 0:
            raise ValueError("cache capacities must be positive")
        device = torch.device(device)
        layers = []
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                kv_heads = config.num_global_key_value_heads
                head_dim = config.global_head_dim
            elif layer_type == "sliding_attention":
                kv_heads = config.num_key_value_heads
                head_dim = config.head_dim
            else:
                raise ValueError(f"unsupported attention type: {layer_type!r}")
            shape = (max_batch_size, kv_heads, max_sequence_length, head_dim)
            layers.append(
                LayerKVCache(
                    key=torch.empty(shape, dtype=dtype, device=device),
                    value=torch.empty(shape, dtype=dtype, device=device),
                    lengths=torch.zeros(
                        max_batch_size, dtype=torch.long, device=device
                    ),
                    max_batch_size=max_batch_size,
                    max_sequence_length=max_sequence_length,
                )
            )
        return cls(layers, max_batch_size, max_sequence_length)

    @property
    def lengths(self) -> torch.Tensor:
        return self.layers[0].lengths

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def device(self) -> torch.device:
        return self.lengths.device

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0].key.dtype

    def reset(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset(batch_size)

    def write(
        self,
        layer_id: int,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.layers[layer_id].write(positions, key, value)

    def view(self, layer_id: int, max_length: int) -> LayerKVView:
        return self.layers[layer_id].view(max_length)

    def commit(self, sequence_lengths: torch.Tensor) -> None:
        for layer in self.layers:
            layer.commit(sequence_lengths)
