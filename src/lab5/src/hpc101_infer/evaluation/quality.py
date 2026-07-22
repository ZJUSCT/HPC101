from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

import torch

from hpc101_infer.engine import InferenceEngine


class Tokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class DatasetNLLResult:
    nll_sum: float
    token_count: int
    sequence_count: int
    mean_nll: float
    perplexity: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_nll(
    engine: InferenceEngine,
    token_sequences: Iterable[Sequence[int]],
    *,
    chunk_size: int = 128,
) -> DatasetNLLResult:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    nll_sum = 0.0
    token_count = 0
    sequence_count = 0
    for sequence_number, token_ids in enumerate(token_sequences, start=1):
        if len(token_ids) < 2:
            raise ValueError(
                f"quality sequence {sequence_number} must contain at least two tokens"
            )
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=engine.device)[None]
        output = engine.score(input_ids, chunk_size=chunk_size)
        nll_sum += output.nll_sum
        token_count += output.token_count
        sequence_count += 1

    if token_count == 0:
        raise ValueError("quality dataset must contain at least one scored token")
    mean_nll = nll_sum / token_count
    return DatasetNLLResult(
        nll_sum=nll_sum,
        token_count=token_count,
        sequence_count=sequence_count,
        mean_nll=mean_nll,
        perplexity=math.exp(mean_nll),
    )


def _normalize_input_ids(input_ids: Any, line_number: int) -> list[int]:
    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        raise ValueError(f"input_ids must be a sequence at line {line_number}")
    if any(
        not isinstance(token_id, Integral) or isinstance(token_id, bool)
        for token_id in input_ids
    ):
        raise ValueError(f"input_ids must contain integers at line {line_number}")
    result = [int(token_id) for token_id in input_ids]
    if len(result) < 2:
        raise ValueError(f"input_ids must contain at least two tokens at line {line_number}")
    return result


def load_quality_token_ids(
    path: str | Path,
    tokenizer: Tokenizer,
    *,
    text_key: str = "text",
    limit: int | None = None,
    add_special_tokens: bool = True,
) -> list[list[int]]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []

    path = Path(path)
    sequences: list[list[int]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"expected an object at line {line_number}")

            if "input_ids" in record:
                token_ids = _normalize_input_ids(record["input_ids"], line_number)
            else:
                text = record.get(text_key)
                if not isinstance(text, str):
                    raise ValueError(
                        f"record at line {line_number} must contain input_ids or "
                        f"a string {text_key!r}"
                    )
                encoded = tokenizer(text, add_special_tokens=add_special_tokens)
                if "input_ids" not in encoded:
                    raise ValueError("tokenizer output does not contain input_ids")
                token_ids = _normalize_input_ids(encoded["input_ids"], line_number)

            sequences.append(token_ids)
            if limit is not None and len(sequences) >= limit:
                break
    return sequences
