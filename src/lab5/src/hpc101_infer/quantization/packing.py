from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from hpc101_infer.quantization.types import QuantizedWeight


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("int4 values must use an integer dtype")
    if values.shape[-1] % 2:
        values = F.pad(values, (0, 1))
    values = values.to(torch.uint8) & 0x0F
    return values[..., 0::2] | (values[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor, length: int | None = None) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError("packed int4 tensor must use torch.uint8")
    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = packed >> 4
    if length is not None:
        if length < 0 or length > unpacked.shape[-1]:
            raise ValueError("invalid unpacked length")
        unpacked = unpacked[..., :length]
    return unpacked


def dequantize_weight(
    quantized: QuantizedWeight,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if quantized.bits != 4:
        raise ValueError("only int4 weights are supported")
    out_features, padded_in_features = quantized.padded_shape
    encoded = unpack_int4(quantized.qweight, padded_in_features)
    encoded = encoded.view(out_features, -1, quantized.group_size).to(dtype)
    scales = quantized.scales.to(dtype).unsqueeze(-1)
    if quantized.symmetric:
        values = encoded - 8.0
    else:
        if quantized.zeros is None:
            raise ValueError("asymmetric quantization requires zero points")
        values = encoded - quantized.zeros.to(dtype).unsqueeze(-1)
    weight = (values * scales).reshape(out_features, padded_in_features)
    return weight[:, : quantized.original_shape[1]].to(dtype)
