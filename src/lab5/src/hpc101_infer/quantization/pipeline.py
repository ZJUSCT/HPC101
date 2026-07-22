from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.layers.linear import BF16LinearFactory
from hpc101_infer.layers.rotary import RotaryEmbedding
from hpc101_infer.models.config import Gemma4TextConfig
from hpc101_infer.models.gemma4 import DecoderLayer
from hpc101_infer.quantization.calibration import ActivationCapture, CalibrationStore
from hpc101_infer.quantization.checkpoint import (
    CheckpointWriter,
    HFSafetensorsSource,
)
from hpc101_infer.quantization.config import QuantizationConfig
from hpc101_infer.quantization.packing import dequantize_weight
from hpc101_infer.quantization.registry import create_quantization_method
from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizationMethod,
    QuantizedModuleManifest,
    QuantizedWeight,
)

__all__ = ["quantize_checkpoint"]

_CHECKPOINT_PREFIX = "model.language_model."
_EXCLUDED_AUXILIARY_FILES = {
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "manifest.json",
    "quantization_config.json",
}


def _parse_target_weight(
    local_key: str,
    target_modules: tuple[str, ...],
) -> tuple[int, str] | None:
    if not local_key.startswith("layers.") or not local_key.endswith(".weight"):
        return None

    parts = local_key.split(".")
    if len(parts) < 4 or parts[-2] not in target_modules:
        return None

    try:
        layer_index = int(parts[1])
    except ValueError:
        return None
    module_name = ".".join(parts[2:]).removesuffix(".weight")
    return layer_index, module_name


def _resolve_tensor(
    module: nn.Module,
    name: str,
) -> tuple[nn.Module, str, torch.Tensor]:
    module_path, separator, tensor_name = name.rpartition(".")
    parent = module.get_submodule(module_path) if separator else module
    if tensor_name in parent._parameters:
        tensor = parent._parameters[tensor_name]
    elif tensor_name in parent._buffers:
        tensor = parent._buffers[tensor_name]
    else:
        raise KeyError(name)
    if tensor is None:
        raise KeyError(name)
    return parent, tensor_name, tensor


def _materialize_tensor(
    module: nn.Module,
    name: str,
    checkpoint_tensor: torch.Tensor,
    device: torch.device,
) -> None:
    parent, tensor_name, target = _resolve_tensor(module, name)
    if checkpoint_tensor.shape != target.shape:
        raise RuntimeError(
            f"shape mismatch for {name}: checkpoint={tuple(checkpoint_tensor.shape)}, "
            f"model={tuple(target.shape)}"
        )
    materialized = checkpoint_tensor.to(device=device)
    if tensor_name in parent._parameters:
        parent._parameters[tensor_name] = nn.Parameter(
            materialized,
            requires_grad=target.requires_grad,
        )
    else:
        parent._buffers[tensor_name] = materialized


class QuantizationPipeline:
    def __init__(
        self,
        source_model_path: str | Path,
        output_dir: str | Path,
        config: QuantizationConfig,
        *,
        method: QuantizationMethod | None,
        calibration_input_ids: torch.Tensor | None,
        calibration_micro_batch_size: int,
        max_calibration_tokens: int,
        print_layer_loss: bool,
        device: str | torch.device,
        max_shard_size_bytes: int,
    ) -> None:
        if calibration_micro_batch_size <= 0:
            raise ValueError("calibration_micro_batch_size must be positive")
        if max_calibration_tokens <= 0:
            raise ValueError("max_calibration_tokens must be positive")
        if calibration_input_ids is not None:
            if calibration_input_ids.ndim != 2:
                raise ValueError("calibration_input_ids must have shape [batch, seq]")
            if (
                calibration_input_ids.shape[0] == 0
                or calibration_input_ids.shape[1] == 0
            ):
                raise ValueError("calibration_input_ids must not be empty")
            if calibration_input_ids.is_floating_point():
                raise TypeError("calibration_input_ids must use an integer dtype")
            calibration_input_ids = (
                calibration_input_ids.detach().cpu().long().contiguous()
            )
        if print_layer_loss and calibration_input_ids is None:
            raise ValueError("print_layer_loss requires calibration_input_ids")

        self.source_model_path = Path(source_model_path)
        self.output_dir = Path(output_dir)
        self.config = config
        self.method = QuantizationPipeline._resolve_method(config, method)
        self.calibration_input_ids = calibration_input_ids
        self.calibration_micro_batch_size = calibration_micro_batch_size
        self.max_calibration_tokens = max_calibration_tokens
        self.print_layer_loss = print_layer_loss
        self.device = torch.device(device)
        self.max_shard_size_bytes = max_shard_size_bytes
        self.source = HFSafetensorsSource(self.source_model_path)
        self.source_keys = frozenset(self.source.keys())
        self.model_config = Gemma4TextConfig.from_pretrained(self.source_model_path)
        self.manifest: dict[str, QuantizedModuleManifest] = {}
        self.layer_metadata: dict[str, dict[str, Any]] = {}

    def run(self) -> dict[str, QuantizedModuleManifest]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        writer = CheckpointWriter(
            self.output_dir,
            max_shard_size_bytes=self.max_shard_size_bytes,
        )
        targets = self._copy_unquantized_tensors(writer)
        if not targets:
            raise RuntimeError("source checkpoint contains no target Linear weights")

        unexpected_layers = sorted(
            set(targets) - set(range(self.model_config.num_hidden_layers))
        )
        if unexpected_layers:
            raise RuntimeError(
                f"target weights reference invalid layers: {unexpected_layers}"
            )

        calibration_store = self._build_initial_calibration_store()
        for layer_index in range(self.model_config.num_hidden_layers):
            target_modules = tuple(sorted(targets.get(layer_index, ())))
            if not target_modules:
                raise RuntimeError(
                    f"layer {layer_index} contains no target Linear weights"
                )
            calibration_store = self._quantize_layer(
                writer,
                layer_index,
                target_modules,
                calibration_store,
            )

        writer.finalize()
        self._write_metadata(self._read_model_config())
        self._copy_auxiliary_files()
        return self.manifest

    def _copy_unquantized_tensors(
        self,
        writer: CheckpointWriter,
    ) -> dict[int, list[str]]:
        targets: dict[int, list[str]] = defaultdict(list)
        for checkpoint_key in self.source.keys():
            if not checkpoint_key.startswith(_CHECKPOINT_PREFIX):
                continue

            local_key = checkpoint_key.removeprefix(_CHECKPOINT_PREFIX)
            target = _parse_target_weight(local_key, self.config.target_modules)
            if target is None:
                writer.add_tensor(local_key, self.source.load_tensor(checkpoint_key))
                continue

            layer_index, module_name = target
            targets[layer_index].append(module_name)
        return targets

    def _build_initial_calibration_store(self) -> CalibrationStore | None:
        if self.calibration_input_ids is None:
            return None
        if (
            self.calibration_input_ids.shape[1]
            > self.model_config.max_position_embeddings
        ):
            raise ValueError(
                "calibration sequence length exceeds model max_position_embeddings"
            )

        embedding_key = f"{_CHECKPOINT_PREFIX}embed_tokens.weight"
        if embedding_key not in self.source_keys:
            raise RuntimeError(f"source checkpoint is missing {embedding_key}")
        is_padding = self.calibration_input_ids == self.model_config.pad_token_id
        sequence_lengths = (~is_padding).sum(dim=1, dtype=torch.long)

        embedding = self.source.load_tensor(embedding_key).to(self.device)
        hidden_states = []
        with torch.inference_mode():
            for start in range(
                0,
                self.calibration_input_ids.shape[0],
                self.calibration_micro_batch_size,
            ):
                input_ids = self.calibration_input_ids[
                    start : start + self.calibration_micro_batch_size
                ].to(self.device)
                batch_hidden_states = F.embedding(input_ids, embedding)
                batch_hidden_states = batch_hidden_states * math.sqrt(
                    self.model_config.hidden_size
                )
                hidden_states.append(batch_hidden_states.cpu())
        return CalibrationStore(torch.cat(hidden_states, dim=0), sequence_lengths)

    def _load_layer(self, layer_index: int) -> DecoderLayer:
        with torch.device("meta"):
            layer = DecoderLayer(
                self.model_config, layer_index, linear_factory=BF16LinearFactory()
            )

        prefix = f"{_CHECKPOINT_PREFIX}layers.{layer_index}."
        for local_name in layer.state_dict():
            checkpoint_key = f"{prefix}{local_name}"
            if checkpoint_key not in self.source_keys:
                raise RuntimeError(f"source checkpoint is missing {checkpoint_key}")
            checkpoint_tensor = self.source.load_tensor(checkpoint_key)
            _materialize_tensor(layer, local_name, checkpoint_tensor, self.device)

        for module in layer.modules():
            if isinstance(module, RotaryEmbedding):
                module.materialize(self.device)
        return layer.eval()

    def _target_linears(
        self,
        layer: DecoderLayer,
        target_modules: tuple[str, ...],
    ) -> dict[str, nn.Linear]:
        modules = dict(layer.named_modules())
        targets: dict[str, nn.Linear] = {}
        for name in target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            targets[name] = module
        return targets

    def _replay_layer(
        self,
        layer: DecoderLayer,
        store: CalibrationStore,
        *,
        collect_outputs: bool,
        capture: ActivationCapture | None = None,
    ) -> CalibrationStore | None:
        outputs = []
        output_sequence_lengths = []
        with torch.inference_mode():
            for batch in store.micro_batches(self.calibration_micro_batch_size):
                hidden_states = batch.hidden_states.to(self.device)
                sequence_lengths = batch.sequence_lengths.to(self.device)
                batch_size, sequence_length = hidden_states.shape[:2]
                position_ids = (
                    torch.arange(sequence_length, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                if capture is not None:
                    capture.set_sequence_lengths(sequence_lengths)
                output = layer(
                    hidden_states,
                    position_ids,
                    sequence_lengths,
                    max_seq_len=sequence_length,
                    kv_cache=None,
                )
                if collect_outputs:
                    outputs.append(output.cpu())
                    output_sequence_lengths.append(batch.sequence_lengths)
        if not collect_outputs:
            return None
        return CalibrationStore(
            torch.cat(outputs, dim=0),
            torch.cat(output_sequence_lengths, dim=0),
        )

    def _apply_fake_quantized_weights(
        self,
        layer: DecoderLayer,
        result: LayerQuantizationResult,
    ) -> None:
        for name, quantized in result.weights.items():
            module = layer.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            fake_quantized = dequantize_weight(
                quantized,
                dtype=module.weight.dtype,
            ).to(self.device)
            module.weight = nn.Parameter(fake_quantized, requires_grad=False)

    @staticmethod
    def _module_quantization_loss(
        module: nn.Linear,
        quantized: QuantizedWeight,
        activations: torch.Tensor,
        *,
        max_chunk_output_elements: int = 1 << 20,
    ) -> float:
        if activations.ndim != 2 or activations.shape[1] != module.in_features:
            raise ValueError("module activations have an invalid shape")
        if max_chunk_output_elements <= 0:
            raise ValueError("max_chunk_output_elements must be positive")

        device = module.weight.device
        dtype = module.weight.dtype
        fake_quantized = dequantize_weight(quantized, dtype=dtype).to(device)
        chunk_tokens = max(1, max_chunk_output_elements // module.out_features)
        squared_error = 0.0
        output_elements = 0

        with torch.inference_mode():
            for start in range(0, activations.shape[0], chunk_tokens):
                inputs = activations[start : start + chunk_tokens].to(
                    device=device,
                    dtype=dtype,
                )
                reference = F.linear(inputs, module.weight, module.bias)
                reconstructed = F.linear(inputs, fake_quantized, module.bias)
                difference = reference.float() - reconstructed.float()
                squared_error += difference.square().sum().item()
                output_elements += difference.numel()

        if output_elements == 0:
            raise ValueError("module activations must not be empty")
        return squared_error / output_elements

    @staticmethod
    def _layer_quantization_loss(
        reference: CalibrationStore,
        quantized: CalibrationStore,
    ) -> float:
        if not torch.equal(reference.sequence_lengths, quantized.sequence_lengths):
            raise ValueError("calibration stores use different sequence lengths")
        positions = torch.arange(reference.hidden_states.shape[1])
        valid = positions.unsqueeze(0) < reference.sequence_lengths.unsqueeze(1)
        return F.mse_loss(
            quantized.hidden_states[valid].float(),
            reference.hidden_states[valid].float(),
        ).item()

    def _quantize_layer(
        self,
        writer: CheckpointWriter,
        layer_index: int,
        target_modules: tuple[str, ...],
        calibration_store: CalibrationStore | None,
    ) -> CalibrationStore | None:
        layer = self._load_layer(layer_index)
        targets = self._target_linears(layer, target_modules)
        activations = None
        next_store = None
        reference_store = None
        has_next_layer = layer_index + 1 < self.model_config.num_hidden_layers

        if calibration_store is not None:
            with ActivationCapture(
                targets,
                max_tokens=self.max_calibration_tokens,
            ) as capture:
                next_store = self._replay_layer(
                    layer,
                    calibration_store,
                    collect_outputs=self.print_layer_loss
                    or (has_next_layer and not self.config.propagate_quantized),
                    capture=capture,
                )
            activations = {name: capture.activations(name) for name in target_modules}
            if self.print_layer_loss:
                reference_store = next_store

        context = LayerContext(
            layer_index=layer_index,
            layer_type=self.model_config.layer_types[layer_index],
            layer=layer,
            activations=activations,
            config=self.config,
            target_modules=target_modules,
        )
        result = self._run_method(context)
        if self.print_layer_loss:
            if activations is None:
                raise RuntimeError("module loss printing requires calibration activations")
            for module_name in target_modules:
                quantized = result.weights[module_name]
                loss = self._module_quantization_loss(
                    targets[module_name],
                    quantized,
                    activations[module_name],
                )
                print(
                    f"layer {layer_index} module {module_name} "
                    f"quantization loss (MSE): {loss:.6e}"
                )
        for module_name, quantized in result.weights.items():
            self._write_quantized_weight(
                writer,
                f"layers.{layer_index}.{module_name}",
                quantized,
            )
        if result.metadata:
            self.layer_metadata[str(layer_index)] = result.metadata

        needs_quantized_outputs = calibration_store is not None and (
            self.print_layer_loss
            or (has_next_layer and self.config.propagate_quantized)
        )
        if needs_quantized_outputs:
            self._apply_fake_quantized_weights(layer, result)
            quantized_store = self._replay_layer(
                layer,
                calibration_store,
                collect_outputs=True,
            )
            if quantized_store is None:
                raise RuntimeError("quantized layer replay produced no outputs")
            if self.print_layer_loss:
                if reference_store is None:
                    raise RuntimeError("reference layer replay produced no outputs")
                loss = self._layer_quantization_loss(
                    reference_store,
                    quantized_store,
                )
                print(f"layer {layer_index} quantization loss (MSE): {loss:.6e}")
            if has_next_layer and self.config.propagate_quantized:
                next_store = quantized_store
        return next_store

    def _write_quantized_weight(
        self,
        writer: CheckpointWriter,
        module_name: str,
        quantized: QuantizedWeight,
    ) -> None:
        qweight_key = f"{module_name}.qweight"
        scales_key = f"{module_name}.scales"
        zeros_key = None if quantized.zeros is None else f"{module_name}.zeros"

        writer.add_tensor(qweight_key, quantized.qweight)
        writer.add_tensor(scales_key, quantized.scales)
        if zeros_key is not None and quantized.zeros is not None:
            writer.add_tensor(zeros_key, quantized.zeros)

        self.manifest[module_name] = QuantizedModuleManifest(
            original_shape=quantized.original_shape,
            padded_shape=quantized.padded_shape,
            qweight_key=qweight_key,
            scales_key=scales_key,
            zeros_key=zeros_key,
            bits=quantized.bits,
            group_size=quantized.group_size,
            symmetric=quantized.symmetric,
            packing=quantized.packing,
        )

    @staticmethod
    def _resolve_method(
        config: QuantizationConfig,
        method: QuantizationMethod | None,
    ) -> QuantizationMethod:
        resolved = method or create_quantization_method(config.algorithm)
        if resolved.name != config.algorithm:
            raise ValueError(
                f"quantization method {resolved.name!r} does not match "
                f"config algorithm {config.algorithm!r}"
            )
        if resolved.version != config.version:
            raise ValueError(
                f"quantization method version {resolved.version!r} does not match "
                f"config version {config.version!r}"
            )
        return resolved

    def _run_method(
        self,
        context: LayerContext,
    ) -> LayerQuantizationResult:
        state = self.method.calibrate_layer(context)
        result = self.method.quantize_layer(context, state)

        expected = set(context.target_modules)
        actual = set(result.weights)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                "quantization method returned an invalid target set: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return result

    def _read_model_config(self) -> dict[str, Any]:
        return json.loads((self.source_model_path / "config.json").read_text())

    def _write_metadata(self, model_config: dict[str, Any]) -> None:
        (self.output_dir / "config.json").write_text(
            json.dumps(model_config, indent=2, sort_keys=True) + "\n"
        )
        (self.output_dir / "quantization_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n"
        )

        raw_manifest: dict[str, Any] = {
            "format_version": 1,
            "method": {
                "name": self.method.name,
                "version": self.method.version,
            },
            "modules": {
                name: entry.to_dict() for name, entry in sorted(self.manifest.items())
            },
        }
        if self.layer_metadata:
            raw_manifest["layer_metadata"] = self.layer_metadata
        (self.output_dir / "manifest.json").write_text(
            json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n"
        )

    def _copy_auxiliary_files(self) -> None:
        for path in self.source_model_path.iterdir():
            if (
                path.is_file()
                and path.name not in _EXCLUDED_AUXILIARY_FILES
                and path.suffix != ".safetensors"
            ):
                shutil.copy2(path, self.output_dir / path.name)


def quantize_checkpoint(
    source_model_path: str | Path,
    output_dir: str | Path,
    config: QuantizationConfig,
    *,
    method: QuantizationMethod | None = None,
    calibration_input_ids: torch.Tensor | None = None,
    calibration_micro_batch_size: int = 1,
    max_calibration_tokens: int = 4096,
    print_layer_loss: bool = False,
    device: str | torch.device = "cpu",
    max_shard_size_bytes: int = 1 << 30,
) -> dict[str, QuantizedModuleManifest]:
    """Quantize a checkpoint, optionally printing per-layer output MSE.

    ``print_layer_loss`` compares each layer's original and fake-quantized
    outputs, so it requires ``calibration_input_ids``.
    """
    return QuantizationPipeline(
        source_model_path,
        output_dir,
        config,
        method=method,
        calibration_input_ids=calibration_input_ids,
        calibration_micro_batch_size=calibration_micro_batch_size,
        max_calibration_tokens=max_calibration_tokens,
        print_layer_loss=print_layer_loss,
        device=device,
        max_shard_size_bytes=max_shard_size_bytes,
    ).run()
