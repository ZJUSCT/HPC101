"""BF16 Linear 与持久化 INT4 权重的参考 Linear 实现。"""

from __future__ import annotations

from typing import Mapping, Protocol

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.packing import dequantize_weight
from hpc101_infer.quantization.types import QuantizedModuleManifest, QuantizedWeight


class LinearFactory(Protocol):
    def create(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module: ...


class BF16LinearFactory:
    def create(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Linear:
        del module_name
        return nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )


class QuantizedLinear(nn.Module):
    """保存 packed INT4 权重，并在 forward 中临时反量化。

    ``qweight`` 使用一个 uint8 保存两个 4-bit code；``scales`` 和可选的
    ``zeros`` 按 ``[out_features, num_groups]`` 存储。输入维度会补齐到
    group size 的整数倍，但矩阵乘前仍只接收原始 ``in_features``。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        *,
        symmetric: bool = True,
        padded_in_features: int | None = None,
        bias: bool = False,
        device: torch.device | str | None = None,
        scale_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if padded_in_features is None:
            padded_in_features = (
                (in_features + group_size - 1) // group_size * group_size
            )
        if padded_in_features < in_features or padded_in_features % group_size:
            raise ValueError("invalid padded input size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.symmetric = symmetric
        self.padded_in_features = padded_in_features
        self.register_buffer(
            "qweight",
            torch.empty(
                out_features,
                padded_in_features // 2,
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                out_features,
                padded_in_features // group_size,
                dtype=scale_dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "zeros",
            (
                None
                if symmetric
                else torch.empty(
                    out_features,
                    padded_in_features // group_size,
                    dtype=torch.uint8,
                    device=device,
                )
            ),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_quantized_weight(
        cls, quantized: QuantizedWeight, bias: torch.Tensor | None = None
    ) -> "QuantizedLinear":
        out_features, in_features = quantized.original_shape
        module = cls(
            in_features,
            out_features,
            quantized.group_size,
            symmetric=quantized.symmetric,
            padded_in_features=quantized.padded_shape[1],
            bias=bias is not None,
            device=quantized.qweight.device,
            scale_dtype=quantized.scales.dtype,
        )
        module.qweight.copy_(quantized.qweight)
        module.scales.copy_(quantized.scales)
        if module.zeros is not None and quantized.zeros is not None:
            module.zeros.copy_(quantized.zeros)
        if bias is not None:
            module.bias.data.copy_(bias)
        return module

    def quantized_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            qweight=self.qweight,
            scales=self.scales,
            zeros=self.zeros,
            original_shape=(self.out_features, self.in_features),
            padded_shape=(self.out_features, self.padded_in_features),
            bits=4,
            group_size=self.group_size,
            symmetric=self.symmetric,
            packing="uint8_little_nibble",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 参考实现允许在单次 forward 内生成高精度临时权重，但不会把完整
        # FP16/BF16 权重持久保存在 module 中。融合 kernel 可替换这一过程。
        weight = dequantize_weight(self.quantized_weight(), dtype=inputs.dtype)
        return F.linear(inputs, weight, self.bias)


class QuantizedLinearFactory:
    def __init__(
        self,
        manifest: Mapping[str, QuantizedModuleManifest],
        *,
        scale_dtype: torch.dtype = torch.float16,
    ) -> None:
        if scale_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("scale_dtype must be float16 or bfloat16")
        self.manifest = dict(manifest)
        self.scale_dtype = scale_dtype
        self._fallback = BF16LinearFactory()

    def create(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module:
        entry = self.manifest.get(module_name)
        if entry is None:
            return self._fallback.create(
                module_name, in_features, out_features, bias, device, dtype
            )
        if entry.original_shape != (out_features, in_features):
            raise RuntimeError(
                f"manifest shape mismatch for {module_name}: "
                f"manifest={entry.original_shape}, model={(out_features, in_features)}"
            )
        scale_dtype = (
            dtype if dtype in {torch.float16, torch.bfloat16} else self.scale_dtype
        )
        return QuantizedLinear(
            in_features,
            out_features,
            entry.group_size,
            symmetric=entry.symmetric,
            padded_in_features=entry.padded_shape[1],
            bias=bias,
            device=device,
            scale_dtype=scale_dtype,
        )
