"""量化校准数据的 CPU 存储、micro-batch 回放和 Linear 输入捕获。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationBatch:
    hidden_states: torch.Tensor
    sequence_lengths: torch.Tensor


class CalibrationStore:
    """在 CPU 保存某一层的输入 hidden states，避免校准集长期占用显存。"""
    def __init__(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        *,
        pin_memory: bool = False,
    ) -> None:
        self.hidden_states = hidden_states.detach().cpu().contiguous()
        self.sequence_lengths = sequence_lengths.detach().cpu().long().contiguous()
        if pin_memory:
            self.hidden_states = self.hidden_states.pin_memory()
            self.sequence_lengths = self.sequence_lengths.pin_memory()

    def micro_batches(self, batch_size: int) -> Iterator[CalibrationBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.hidden_states.shape[0], batch_size):
            end = start + batch_size
            yield CalibrationBatch(
                hidden_states=self.hidden_states[start:end],
                sequence_lengths=self.sequence_lengths[start:end],
            )

    def __len__(self) -> int:
        return self.hidden_states.shape[0]


class ActivationCapture:
    """通过 forward pre-hook 收集目标 Linear 的有效输入 token。

    捕获结果统一展平为 ``[valid_tokens, in_features]``。``max_tokens``
    限制每个模块保留的 token 数，防止长校准集耗尽主机内存。
    """
    def __init__(
        self,
        modules: dict[str, torch.nn.Module],
        *,
        max_tokens: int,
        pin_memory: bool = False,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.modules = dict(modules)
        self.max_tokens = max_tokens
        self.pin_memory = pin_memory
        self._captured: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
        self._token_counts = {name: 0 for name in modules}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._sequence_lengths: torch.Tensor | None = None

    def set_sequence_lengths(self, sequence_lengths: torch.Tensor) -> None:
        self._sequence_lengths = sequence_lengths.detach().long()

    def _hook(self, name: str):
        def capture(module, args) -> None:
            del module
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError("captured module input must begin with a tensor")
            remaining = self.max_tokens - self._token_counts[name]
            if remaining <= 0:
                return

            inputs = args[0].detach()
            if self._sequence_lengths is None:
                values = inputs.reshape(-1, inputs.shape[-1])
            else:
                # 只保留真实 token；右侧 padding 不应参与激活统计。
                sequence_lengths = self._sequence_lengths.to(inputs.device)
                positions = torch.arange(inputs.shape[1], device=inputs.device)
                valid = positions.unsqueeze(0) < sequence_lengths.unsqueeze(1)
                values = inputs[valid]

            values = values[:remaining].cpu().contiguous()
            if self.pin_memory:
                values = values.pin_memory()
            self._captured[name].append(values)
            self._token_counts[name] += values.shape[0]

        return capture

    def __enter__(self) -> "ActivationCapture":
        if self._handles:
            raise RuntimeError("activation capture is already active")
        self._handles = [
            module.register_forward_pre_hook(self._hook(name))
            for name, module in self.modules.items()
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._sequence_lengths = None

    def activations(self, name: str) -> torch.Tensor:
        if name not in self._captured:
            raise KeyError(name)
        tensors = self._captured[name]
        if not tensors:
            raise RuntimeError(f"no activations captured for {name}")
        return torch.cat(tensors, dim=0)
