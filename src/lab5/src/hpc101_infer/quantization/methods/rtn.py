"""Round-to-Nearest INT4 基线，用于验证 packing 与推理闭环。"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizedWeight,
    SCALE_DTYPES,
)

from hpc101_infer.quantization.packing import pack_int4


def quantize_weight_rtn(
    weight: torch.Tensor,
    group_size: int,
    *,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> QuantizedWeight:
    """逐输出通道、逐 group 量化权重，不使用任何校准激活。

    RTN 是便于调试的参考基线：它验证 group padding、scale/zero point、
    INT4 packing 和 checkpoint 加载，但不补偿量化误差。
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if scale_dtype not in SCALE_DTYPES.values():
        raise ValueError("unsupported scale dtype")

    weight = weight.detach()
    out_features, in_features = weight.shape
    padded_in_features = math.ceil(in_features / group_size) * group_size
    # 仅在输入维补零，使每个输出通道都能重排为完整的量化 groups。
    padded = F.pad(weight.float(), (0, padded_in_features - in_features))
    groups = padded.view(out_features, -1, group_size)

    if symmetric:
        scales = groups.abs().amax(dim=-1) / 7.0
        scales = scales.clamp_min(torch.finfo(torch.float32).eps)
        quantized = torch.round(groups / scales.unsqueeze(-1)).clamp(-8, 7)
        encoded = (quantized.to(torch.int16) + 8).to(torch.uint8)
        zeros = None
    else:
        zero = torch.zeros_like(groups[..., 0])
        minimum = torch.minimum(groups.amin(dim=-1), zero)
        maximum = torch.maximum(groups.amax(dim=-1), zero)
        scales = ((maximum - minimum) / 15.0).clamp_min(torch.finfo(torch.float32).eps)
        zeros = torch.round(-minimum / scales).clamp(0, 15).to(torch.uint8)
        encoded = torch.round(groups / scales.unsqueeze(-1))
        encoded = (encoded + zeros.unsqueeze(-1)).clamp(0, 15).to(torch.uint8)

    return QuantizedWeight(
        qweight=pack_int4(encoded.reshape(out_features, padded_in_features)),
        scales=scales.to(scale_dtype),
        zeros=zeros,
        original_shape=(out_features, in_features),
        padded_shape=(out_features, padded_in_features),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        packing="uint8_little_nibble",
    )


class RTNQuantizationMethod:
    name = "rtn"
    version = "1"

    def calibrate_layer(self, context: LayerContext) -> None:
        del context
        return None

    def quantize_layer(
        self, context: LayerContext, state: None
    ) -> LayerQuantizationResult:
        del state
        scale_dtype = SCALE_DTYPES[context.config.scale_dtype]
        weights = {}
        modules = dict(context.layer.named_modules())
        for name in context.target_modules:
            module = modules[name]
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            weights[name] = quantize_weight_rtn(
                module.weight,
                context.config.group_size,
                symmetric=context.config.symmetric,
                scale_dtype=scale_dtype,
            )
        return LayerQuantizationResult(weights=weights, metadata={})
