from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.packing import pack_int4
from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizedWeight,
    SCALE_DTYPES,
)


@dataclass(frozen=True)
class GPTQOptions:
    block_size: int
    damp_percent: float


@dataclass(frozen=True)
class GPTQModuleState:
    activations: torch.Tensor
    activation_tokens: int


@dataclass(frozen=True)
class GPTQLayerState:
    modules: Mapping[str, GPTQModuleState]
    options: GPTQOptions


def _calibration_option(
    calibration: Mapping[str, Any],
    name: str,
    default: int | float,
) -> int | float:
    gptq = calibration.get("gptq", calibration)
    if not isinstance(gptq, Mapping):
        raise TypeError("config.calibration['gptq'] must be a mapping")
    return gptq.get(name, default)


def _parse_gptq_options(calibration: Mapping[str, Any]) -> GPTQOptions:
    block_size = _calibration_option(calibration, "block_size", 128)
    damp_percent = _calibration_option(calibration, "damp_percent", 0.01)

    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("GPTQ block_size must be a positive integer")
    if (
        not isinstance(damp_percent, (int, float))
        or isinstance(damp_percent, bool)
        or not 0.0 < float(damp_percent) < 1.0
    ):
        raise ValueError("GPTQ damp_percent must be in (0, 1)")

    gptq = calibration.get("gptq", calibration)
    if isinstance(gptq, Mapping) and gptq.get("desc_act", False):
        raise ValueError("GPTQ desc_act is not supported by this checkpoint format")
    return GPTQOptions(
        block_size=block_size,
        damp_percent=float(damp_percent),
    )


# STUDENT IMPLEMENTATION START: Hessian construction.
# This function is a suitable public TODO in the student starter repository.
def _build_hessian(
    activations: torch.Tensor,
    padded_in_features: int,
) -> torch.Tensor:
    if activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("activations must be a floating-point matrix")
    if activations.shape[0] == 0:
        raise ValueError("activations must not be empty")
    if padded_in_features < activations.shape[1]:
        raise ValueError("padded input size is smaller than the activation width")
    if not torch.isfinite(activations).all():
        raise ValueError("activations must contain only finite values")

    values = activations.detach().float()
    values = F.pad(values, (0, padded_in_features - values.shape[1]))
    return values.t().matmul(values).mul_(2.0 / values.shape[0])


# STUDENT IMPLEMENTATION START: dead-column handling, damping, and inverse
# Hessian factorization. This function is a suitable public TODO.
def _inverse_hessian_factor(
    hessian: torch.Tensor,
    damp_percent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix")
    if not hessian.is_floating_point():
        raise ValueError("hessian must use a floating-point dtype")
    if not 0.0 < damp_percent < 1.0:
        raise ValueError("damp_percent must be in (0, 1)")
    if not torch.isfinite(hessian).all():
        raise ValueError("hessian must contain only finite values")

    stabilized = hessian.detach().float().clone()
    diagonal = stabilized.diagonal()
    dead_columns = diagonal == 0
    diagonal[dead_columns] = 1.0
    damping = damp_percent * diagonal.mean()
    diagonal.add_(damping)

    try:
        factor = torch.linalg.cholesky(stabilized)
        inverse = torch.cholesky_inverse(factor)
        inverse_factor = torch.linalg.cholesky(inverse, upper=True)
    except RuntimeError as error:
        raise RuntimeError(
            "GPTQ Hessian is not positive definite after damping; "
            "increase calibration.gptq.damp_percent"
        ) from error
    return inverse_factor, dead_columns


# STUDENT IMPLEMENTATION START: per-output-channel group quantization
# parameters. This function is a suitable public TODO.
def _find_group_qparams(
    weight_group: torch.Tensor,
    *,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if weight_group.ndim != 2 or not weight_group.is_floating_point():
        raise ValueError("weight_group must be a floating-point matrix")
    if weight_group.shape[1] == 0:
        raise ValueError("weight_group must not be empty")
    if not torch.isfinite(weight_group).all():
        raise ValueError("weight_group must contain only finite values")

    epsilon = torch.finfo(torch.float32).eps
    values = weight_group.float()
    if symmetric:
        scales = (values.abs().amax(dim=1) / 7.0).clamp_min(epsilon)
        return scales, None

    zero = torch.zeros_like(values[:, 0])
    minimum = torch.minimum(values.amin(dim=1), zero)
    maximum = torch.maximum(values.amax(dim=1), zero)
    scales = ((maximum - minimum) / 15.0).clamp_min(epsilon)
    zeros = torch.round(-minimum / scales).clamp(0, 15).to(torch.uint8)
    return scales, zeros


# STUDENT IMPLEMENTATION START: quantize one corrected GPTQ column while
# retaining both the dequantized value and exact int4 code.
def _quantize_column(
    column: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    *,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if column.ndim != 1 or scales.ndim != 1 or column.shape != scales.shape:
        raise ValueError("column and scales must be matching vectors")

    if symmetric:
        if zeros is not None:
            raise ValueError("symmetric quantization does not use zero points")
        quantized = torch.round(column / scales).clamp(-8, 7)
        encoded = (quantized.to(torch.int16) + 8).to(torch.uint8)
        return quantized * scales, encoded

    if zeros is None or zeros.ndim != 1 or zeros.shape != column.shape:
        raise ValueError("asymmetric quantization requires matching zero points")
    encoded = torch.round(column / scales) + zeros.to(column.dtype)
    encoded = encoded.clamp(0, 15).to(torch.uint8)
    dequantized = (encoded.to(column.dtype) - zeros.to(column.dtype)) * scales
    return dequantized, encoded


def _current_group_weight(
    weight: torch.Tensor,
    block_weight: torch.Tensor,
    *,
    group_start: int,
    group_end: int,
    block_start: int,
    block_end: int,
) -> torch.Tensor:
    group = weight[:, group_start:group_end].clone()
    overlap_start = max(group_start, block_start)
    overlap_end = min(group_end, block_end)
    if overlap_start < overlap_end:
        group[:, overlap_start - group_start : overlap_end - group_start] = (
            block_weight[
                :,
                overlap_start - block_start : overlap_end - block_start,
            ]
        )
    return group


# STUDENT IMPLEMENTATION START: complete GPTQ Core. The starter assignment can
# keep validation/output assembly and replace the Hessian, qparam, and block-wise
# error-compensation portions with TODOs.
def quantize_weight_gptq(
    weight: torch.Tensor,
    activations: torch.Tensor,
    group_size: int,
    *,
    block_size: int = 128,
    damp_percent: float = 0.01,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[QuantizedWeight, dict[str, float | int]]:
    """
    使用 GPTQ 算法将权重量化为 INT4。

    参数：
        weight: 待量化的权重张量，形状为 (out_features, in_features)。
        activations: 校准数据集的输入激活，形状为 (calibration_tokens, in_features)。
        group_size: 量化粒度，即每个 group 中的列数。
        block_size: 分块计算时每个 block 中的列数。
        damp_percent: 阻尼比例，用于改善 Hessian 的数值稳定性。
        symmetric: 是否使用对称量化。
        scale_dtype: 缩放因子的 dtype。

    返回：
        quantized_weight: 量化后的权重对象，具体详见 QuantizedWeight 类的定义。
        metadatas: 量化过程中的统计信息，不影响评测，用于分析和调试。
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("activations must be a floating-point matrix")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation width does not match weight input size")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < damp_percent < 1.0:
        raise ValueError("damp_percent must be in (0, 1)")
    if scale_dtype not in SCALE_DTYPES.values():
        raise ValueError("unsupported scale dtype")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values")

    # STUDENT IMPLEMENTATION START: GPTQ Core.

    out_features, in_features = weight.shape
    padded_in_features = math.ceil(in_features / group_size) * group_size
    working = F.pad(
        weight.detach().float(),
        (0, padded_in_features - in_features),
    )
    hessian = _build_hessian(activations, padded_in_features).to(working.device)
    inverse_factor, dead_columns = _inverse_hessian_factor(
        hessian,
        damp_percent,
    )
    working[:, dead_columns] = 0.0

    encoded = torch.empty(
        (out_features, padded_in_features),
        dtype=torch.uint8,
        device=working.device,
    )
    group_count = padded_in_features // group_size
    scales = torch.empty(
        (out_features, group_count),
        dtype=torch.float32,
        device=working.device,
    )
    zeros = None
    if not symmetric:
        zeros = torch.empty(
            (out_features, group_count),
            dtype=torch.uint8,
            device=working.device,
        )

    current_scales: torch.Tensor | None = None
    current_zeros: torch.Tensor | None = None
    predicted_loss = torch.zeros((), dtype=torch.float32, device=working.device)

    for block_start in range(0, padded_in_features, block_size):
        block_end = min(block_start + block_size, padded_in_features)
        block_weight = working[:, block_start:block_end].clone()
        block_errors = torch.empty_like(block_weight)

        for local_index in range(block_end - block_start):
            column_index = block_start + local_index
            if column_index % group_size == 0:
                group_index = column_index // group_size
                group_end = min(column_index + group_size, padded_in_features)
                group_weight = _current_group_weight(
                    working,
                    block_weight,
                    group_start=column_index,
                    group_end=group_end,
                    block_start=block_start,
                    block_end=block_end,
                )
                current_scales, current_zeros = _find_group_qparams(
                    group_weight,
                    symmetric=symmetric,
                )
                scales[:, group_index] = current_scales
                if zeros is not None:
                    if current_zeros is None:
                        raise RuntimeError("missing asymmetric zero points")
                    zeros[:, group_index] = current_zeros

            if current_scales is None:
                raise RuntimeError("GPTQ group parameters were not initialized")
            column = block_weight[:, local_index]
            dequantized, column_codes = _quantize_column(
                column,
                current_scales,
                current_zeros,
                symmetric=symmetric,
            )
            encoded[:, column_index] = column_codes

            diagonal = inverse_factor[column_index, column_index]
            error = (column - dequantized) / diagonal
            predicted_loss.add_(0.5 * error.square().sum())
            block_weight[:, local_index:] -= error.unsqueeze(1).matmul(
                inverse_factor[
                    column_index,
                    column_index:block_end,
                ].unsqueeze(0)
            )
            block_errors[:, local_index] = error

        if block_end < padded_in_features:
            working[:, block_end:] -= block_errors.matmul(
                inverse_factor[block_start:block_end, block_end:]
            )

    # Student implementation end: GPTQ Core.

    quantized = QuantizedWeight(
        qweight=pack_int4(encoded),
        scales=scales.to(scale_dtype),
        zeros=zeros,
        original_shape=(out_features, in_features),
        padded_shape=(out_features, padded_in_features),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        packing="uint8_little_nibble",
    )
    metadata: dict[str, float | int] = {
        "activation_tokens": activations.shape[0],
        "block_size": block_size,
        "damp_percent": damp_percent,
        "dead_columns": int(dead_columns.sum().item()),
        "predicted_loss": predicted_loss.item() / activations.shape[0],
    }
    return quantized, metadata


class GPTQQuantizationMethod:
    name = "gptq"
    version = "1"

    # FRAMEWORK-PROVIDED: validates and stores captured activations. This is
    # interface plumbing rather than part of the student GPTQ algorithm.
    def calibrate_layer(self, context: LayerContext) -> GPTQLayerState:
        if context.activations is None:
            raise ValueError("GPTQ requires calibration activations")

        options = _parse_gptq_options(context.config.calibration or {})
        modules = dict(context.layer.named_modules())
        states: dict[str, GPTQModuleState] = {}
        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            if name not in context.activations:
                raise ValueError(f"missing calibration activations for {name}")

            activations = context.activations[name]
            if activations.ndim != 2 or activations.shape[1] != module.in_features:
                raise ValueError(
                    f"invalid calibration activation shape for {name}: "
                    f"expected [tokens, {module.in_features}], "
                    f"got {tuple(activations.shape)}"
                )
            if activations.shape[0] == 0:
                raise ValueError(f"calibration activations for {name} are empty")
            if not activations.is_floating_point():
                raise TypeError(f"calibration activations for {name} must be floating")
            if not torch.isfinite(activations).all():
                raise ValueError(f"calibration activations for {name} are not finite")

            states[name] = GPTQModuleState(
                activations=activations.detach(),
                activation_tokens=activations.shape[0],
            )

        return GPTQLayerState(modules=states, options=options)

    # FRAMEWORK-PROVIDED: connects the student numerical kernel to the common
    # QuantizationMethod interface and checkpoint metadata.
    def quantize_layer(
        self, context: LayerContext, state: GPTQLayerState
    ) -> LayerQuantizationResult:
        if not isinstance(state, GPTQLayerState):
            raise TypeError("state must be a GPTQLayerState")

        scale_dtype = SCALE_DTYPES[context.config.scale_dtype]
        modules = dict(context.layer.named_modules())
        weights: dict[str, QuantizedWeight] = {}
        module_metadata: dict[str, dict[str, float | int]] = {}

        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            module_state = state.modules.get(name)
            if module_state is None:
                raise ValueError(f"missing GPTQ calibration state for {name}")

            try:
                weights[name], module_metadata[name] = quantize_weight_gptq(
                    module.weight,
                    module_state.activations,
                    context.config.group_size,
                    block_size=state.options.block_size,
                    damp_percent=state.options.damp_percent,
                    symmetric=context.config.symmetric,
                    scale_dtype=scale_dtype,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"GPTQ failed for layer {context.layer_index} module {name} "
                    f"with {module_state.activation_tokens} calibration tokens: {error}"
                ) from error

        return LayerQuantizationResult(
            weights=weights,
            metadata={
                "gptq": {
                    "block_size": state.options.block_size,
                    "damp_percent": state.options.damp_percent,
                    "modules": module_metadata,
                }
            },
        )
