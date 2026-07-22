from __future__ import annotations

from torch import nn

TARGET_LINEAR_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def gemma4_target_linears(model: nn.Module) -> dict[str, nn.Linear]:
    targets = {}
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and name.startswith("layers.")
            and name.rsplit(".", 1)[-1] in TARGET_LINEAR_NAMES
        ):
            targets[name] = module
    return targets
