from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

import torch

from hpc101_infer.data.jsonl import load_jsonl_texts


class Tokenizer(Protocol):
    def __call__(self, texts: str | list[str], **kwargs: Any) -> Mapping[str, Any]: ...


def load_token_ids(
    source: str | Path,
    tokenizer: Tokenizer,
    *,
    text_key: str = "text",
    limit: int | None = None,
    **tokenizer_kwargs: Any,
) -> list[list[int]]:
    """Tokenize one JSON object or every record in a JSONL file.

    The outer list always corresponds to input records, including when ``source``
    is a single mapping. No padding is added, so each inner list keeps its natural
    sequence length.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []

    texts = load_jsonl_texts(source, text_key=text_key, limit=limit)

    token_id_sequences: list[list[int]] = []
    for record_number, text in enumerate(texts, start=1):
        encoded = tokenizer(text, **tokenizer_kwargs)
        if "input_ids" not in encoded:
            raise ValueError("tokenizer output does not contain input_ids")

        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
            raise ValueError("tokenizer input_ids must be a one-dimensional sequence")
        if not input_ids:
            raise ValueError(
                f"tokenizer returned empty input_ids for record {record_number}"
            )
        if any(
            not isinstance(token_id, Integral) or isinstance(token_id, bool)
            for token_id in input_ids
        ):
            raise TypeError("tokenizer input_ids must contain only integers")
        token_id_sequences.append([int(token_id) for token_id in input_ids])
    return token_id_sequences


def texts_to_token_ids(
    texts: Sequence[str],
    tokenizer: Tokenizer,
) -> torch.Tensor:
    if not texts:
        raise ValueError("texts must not be empty")

    encoded = tokenizer(list(texts), padding=True, return_tensors="pt")
    if "input_ids" not in encoded:
        raise ValueError("tokenizer output does not contain input_ids")

    input_ids = torch.as_tensor(encoded["input_ids"])
    if input_ids.ndim != 2:
        raise ValueError("tokenizer input_ids must have shape [batch, seq]")
    if input_ids.shape[0] != len(texts) or input_ids.shape[1] == 0:
        raise ValueError("tokenizer input_ids have an invalid shape")
    if input_ids.is_floating_point():
        raise TypeError("tokenizer input_ids must use an integer dtype")
    return input_ids.long().contiguous()
