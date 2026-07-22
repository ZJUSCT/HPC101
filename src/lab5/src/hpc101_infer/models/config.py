from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RotaryConfig:
    rope_type: str
    rope_theta: float
    partial_rotary_factor: float | None = None
    factor: float = 1.0

    def get(self, key: str, default: float | None = None) -> float | None:
        return getattr(self, key, default)


@dataclass(frozen=True)
class Gemma4TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    layer_types: tuple[str, ...]
    sliding_window: int
    max_position_embeddings: int
    rms_norm_eps: float
    hidden_activation: str
    final_logit_softcapping: float | None
    rope_parameters: dict[str, dict[str, Any]]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    attention_k_eq_v: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Gemma4TextConfig":
        raw = json.loads((Path(model_path) / "config.json").read_text())
        text = raw.get("text_config", raw)
        if text.get("model_type") != "gemma4_unified_text":
            raise ValueError(f"unsupported model_type: {text.get('model_type')!r}")

        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            num_global_key_value_heads=text["num_global_key_value_heads"],
            head_dim=text["head_dim"],
            global_head_dim=text["global_head_dim"],
            layer_types=tuple(text["layer_types"]),
            sliding_window=text["sliding_window"],
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=text["rms_norm_eps"],
            hidden_activation=text["hidden_activation"],
            final_logit_softcapping=text.get("final_logit_softcapping"),
            rope_parameters=text["rope_parameters"],
            pad_token_id=text["pad_token_id"],
            bos_token_id=text["bos_token_id"],
            eos_token_id=text["eos_token_id"],
            attention_k_eq_v=text.get("attention_k_eq_v", True),
        )

    def get_rope_config(self, layer_type: str) -> RotaryConfig:
        params = self.rope_parameters[layer_type]
        return RotaryConfig(
            rope_type=params["rope_type"],
            rope_theta=params["rope_theta"],
            partial_rotary_factor=params.get("partial_rotary_factor"),
            factor=params.get("factor", 1.0),
        )
