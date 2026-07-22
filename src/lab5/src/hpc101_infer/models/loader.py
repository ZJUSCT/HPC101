"""从 Hugging Face safetensors checkpoint 流式 materialize Gemma 4。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from torch import nn

from hpc101_infer.layers.linear import QuantizedLinearFactory
from hpc101_infer.layers.rotary import RotaryEmbedding
from hpc101_infer.models.config import Gemma4TextConfig
from hpc101_infer.models.gemma4 import Gemma4ForCausalLM
from hpc101_infer.quantization.checkpoint import (
    HFSafetensorsSource,
    QuantizedCheckpointSource,
)
from hpc101_infer.quantization.types import SCALE_DTYPES


def _resolve_tensor(model: nn.Module, name: str) -> tuple[nn.Module, str, torch.Tensor]:
    module_path, separator, tensor_name = name.rpartition(".")
    module = model.get_submodule(module_path) if separator else model
    if tensor_name in module._parameters:
        tensor = module._parameters[tensor_name]
    elif tensor_name in module._buffers:
        tensor = module._buffers[tensor_name]
    else:
        raise KeyError(name)
    if tensor is None:
        raise KeyError(name)
    return module, tensor_name, tensor


def _materialize_tensor(
    model: nn.Module,
    name: str,
    checkpoint_tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    preserve_target_dtype: bool = False,
) -> None:
    module, tensor_name, target = _resolve_tensor(model, name)
    if checkpoint_tensor.shape != target.shape:
        raise RuntimeError(
            f"shape mismatch for {name}: checkpoint={tuple(checkpoint_tensor.shape)}, "
            f"model={tuple(target.shape)}"
        )
    target_dtype = (
        target.dtype
        if preserve_target_dtype or not checkpoint_tensor.is_floating_point()
        else dtype
    )
    materialized = checkpoint_tensor.to(device=device, dtype=target_dtype)
    if tensor_name in module._parameters:
        module._parameters[tensor_name] = nn.Parameter(
            materialized, requires_grad=target.requires_grad
        )
    else:
        module._buffers[tensor_name] = materialized


def _quantized_key_map(source: QuantizedCheckpointSource) -> dict[str, str]:
    result: dict[str, str] = {}
    for module_name, entry in source.manifest.items():
        result[entry.qweight_key] = f"{module_name}.qweight"
        result[entry.scales_key] = f"{module_name}.scales"
        if entry.zeros_key is not None:
            result[entry.zeros_key] = f"{module_name}.zeros"
    return result


def load_gemma4(
    model_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_position_embeddings: int | None = None,
    linear_backend: str = "bf16",
) -> Gemma4ForCausalLM:
    """在 meta device 构造模型，再逐 tensor 加载到目标设备。

    该过程避免先在 CPU 上构造完整 ``state_dict``。量化 checkpoint 会根据
    manifest 创建 ``QuantizedLinear``，普通 checkpoint 则创建 BF16 Linear。
    """
    model_path = Path(model_path)
    device = torch.device(device)
    config = Gemma4TextConfig.from_pretrained(model_path)
    if max_position_embeddings is not None:
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if max_position_embeddings > config.max_position_embeddings:
            raise ValueError("runtime max_position_embeddings exceeds model capacity")
        config = replace(config, max_position_embeddings=max_position_embeddings)

    is_quantized = (model_path / "quantization_config.json").is_file()
    if linear_backend not in {"bf16", "int4_reference"}:
        raise ValueError(f"unsupported linear backend: {linear_backend}")
    if linear_backend == "int4_reference" and not is_quantized:
        raise ValueError("int4_reference requires a quantized checkpoint")

    if is_quantized:
        source = QuantizedCheckpointSource(model_path)
        scale_dtype = SCALE_DTYPES[source.quantization_config.scale_dtype]
        linear_factory = QuantizedLinearFactory(
            source.manifest, scale_dtype=scale_dtype
        )
        source_to_local = _quantized_key_map(source)
        prefix = ""
    else:
        source = HFSafetensorsSource(model_path)
        linear_factory = None
        source_to_local = {}
        prefix = "model.language_model."

    # meta model 只记录结构和 shape，不为完整模型分配真实权重存储。
    with torch.device("meta"):
        model = Gemma4ForCausalLM(config, linear_factory=linear_factory)

    expected = set(model.state_dict())
    loaded: set[str] = set()
    unexpected: set[str] = set()
    duplicated: set[str] = set()

    for checkpoint_key in source.keys():
        if prefix and not checkpoint_key.startswith(prefix):
            continue
        local_key = source_to_local.get(
            checkpoint_key,
            checkpoint_key[len(prefix) :] if prefix else checkpoint_key,
        )
        if local_key in loaded:
            duplicated.add(local_key)
            continue
        if local_key not in expected:
            unexpected.add(local_key)
            continue
        checkpoint_tensor = source.load_tensor(checkpoint_key)
        _materialize_tensor(
            model,
            local_key,
            checkpoint_tensor,
            device,
            dtype,
            preserve_target_dtype=is_quantized and local_key.endswith(".scales"),
        )
        loaded.add(local_key)
        del checkpoint_tensor

    missing = expected - loaded
    if missing or unexpected or duplicated:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"duplicated={sorted(duplicated)}"
        )

    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.materialize(device)
    return model.eval()
