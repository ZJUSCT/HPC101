from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from hpc101_infer.quantization.config import QuantizationConfig
from hpc101_infer.quantization.types import (
    QuantizedModuleManifest,
    TensorMetadata,
)


class CheckpointSource(Protocol):
    def keys(self) -> Iterable[str]: ...

    def metadata(self, name: str) -> TensorMetadata: ...

    def load_tensor(self, name: str) -> torch.Tensor: ...


def checkpoint_weight_map(model_path: str | Path) -> dict[str, str]:
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        raw = json.loads(index_path.read_text())
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"invalid checkpoint index: {index_path}")
        missing = sorted(
            {
                shard
                for shard in weight_map.values()
                if not (model_path / shard).is_file()
            }
        )
        if missing:
            raise RuntimeError(f"checkpoint shards not found: {missing}")
        return dict(weight_map)

    checkpoint = model_path / "model.safetensors"
    if checkpoint.is_file():
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            return {key: checkpoint.name for key in handle.keys()}
    raise FileNotFoundError(f"no safetensors checkpoint found in {model_path}")


class HFSafetensorsSource:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        declared = checkpoint_weight_map(self.model_path)
        self._weight_map: dict[str, str] = {}
        for shard in dict.fromkeys(declared.values()):
            with safe_open(
                self.model_path / shard, framework="pt", device="cpu"
            ) as handle:
                for key in handle.keys():
                    if key in self._weight_map:
                        raise RuntimeError(f"duplicated checkpoint tensor: {key}")
                    self._weight_map[key] = shard
        declared_keys = set(declared)
        actual_keys = set(self._weight_map)
        if declared_keys != actual_keys:
            raise RuntimeError(
                "checkpoint index mismatch: "
                f"missing={sorted(declared_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - declared_keys)}"
            )

    def keys(self) -> tuple[str, ...]:
        return tuple(self._weight_map)

    def metadata(self, name: str) -> TensorMetadata:
        try:
            shard = self._weight_map[name]
        except KeyError as error:
            raise KeyError(name) from error
        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "F64": torch.float64,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(name)
            return TensorMetadata(
                tuple(tensor_slice.get_shape()),
                dtype_map[tensor_slice.get_dtype()],
                shard,
            )

    def load_tensor(self, name: str) -> torch.Tensor:
        try:
            shard = self._weight_map[name]
        except KeyError as error:
            raise KeyError(name) from error
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


class QuantizedCheckpointSource(HFSafetensorsSource):
    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        manifest_path = self.model_path / "manifest.json"
        config_path = self.model_path / "quantization_config.json"
        if not manifest_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("quantized checkpoint metadata is incomplete")
        raw_manifest = json.loads(manifest_path.read_text())
        modules = raw_manifest.get("modules")
        if not isinstance(modules, dict):
            raise RuntimeError("invalid quantized checkpoint manifest")
        self.manifest = {
            name: QuantizedModuleManifest.from_dict(entry)
            for name, entry in modules.items()
        }
        self.quantization_config = QuantizationConfig.from_dict(
            json.loads(config_path.read_text())
        )


class CheckpointWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_shard_size_bytes: int = 1 << 30,
    ) -> None:
        if max_shard_size_bytes <= 0:
            raise ValueError("max_shard_size_bytes must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_size_bytes = max_shard_size_bytes
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._weight_map: dict[str, str] = {}
        self._temporary_shards: list[Path] = []

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._weight_map or name in self._buffer:
            raise RuntimeError(f"duplicated output tensor: {name}")
        tensor = tensor.detach().cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        if self._buffer and self._buffer_bytes + size > self.max_shard_size_bytes:
            self._flush()
        self._buffer[name] = tensor
        self._buffer_bytes += size

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard_index = len(self._temporary_shards) + 1
        path = self.output_dir / f"model-{shard_index:05d}.safetensors"
        save_file(self._buffer, path)
        for name in self._buffer:
            self._weight_map[name] = path.name
        self._temporary_shards.append(path)
        self._buffer = {}
        self._buffer_bytes = 0

    def finalize(self) -> dict[str, str]:
        self._flush()
        if not self._temporary_shards:
            raise RuntimeError("cannot finalize an empty checkpoint")
        shard_count = len(self._temporary_shards)
        renamed: dict[str, str] = {}
        for index, old_path in enumerate(self._temporary_shards, 1):
            new_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            new_path = self.output_dir / new_name
            old_path.replace(new_path)
            renamed[old_path.name] = new_name
        self._weight_map = {
            key: renamed[shard] for key, shard in self._weight_map.items()
        }
        total_size = sum(
            (self.output_dir / shard).stat().st_size
            for shard in set(self._weight_map.values())
        )
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": self._weight_map,
        }
        (self.output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        return dict(self._weight_map)
