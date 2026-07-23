"""面向教学的同步推理引擎，显式展示 prefill、decode 和 KV cache 数据流。"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F

from hpc101_infer.config import EngineConfig
from hpc101_infer.models.gemma4 import Gemma4ForCausalLM
from hpc101_infer.models.loader import load_gemma4
from hpc101_infer.runtime.batch import Batch
from hpc101_infer.runtime.kv_cache import KVCache
from hpc101_infer.runtime.metrics import measure_operation
from hpc101_infer.sampling import Sampler
from hpc101_infer.scheduler import RequestState, create_scheduler
from hpc101_infer.types import (
    DecodeOutput,
    GenerationOutput,
    GenerationRequest,
    PrefillOutput,
    RequestMetrics,
    ScoreOutput,
)


class InferenceEngine:
    def __init__(
        self,
        model: Gemma4ForCausalLM,
        config: EngineConfig,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        first_parameter = next(model.parameters())
        if (
            first_parameter.device != self.device
            or first_parameter.dtype != config.dtype
        ):
            model = model.to(device=self.device, dtype=config.dtype)
        self.model = model.eval()
        self.cache = KVCache.allocate(
            model.config,
            config.max_batch_size,
            config.max_sequence_length,
            config.dtype,
            self.device,
        )
        self.sampler = Sampler(self.device, self.model.config.vocab_size)
        self._batch_size = 0
        torch.manual_seed(config.seed)

    @classmethod
    def from_pretrained(
        cls, model_path: str, config: EngineConfig
    ) -> "InferenceEngine":
        from transformers import AutoTokenizer

        model = load_gemma4(
            model_path,
            device=config.device,
            dtype=config.dtype,
            max_position_embeddings=config.max_sequence_length,
            linear_backend=config.linear_backend,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(model, config, tokenizer)

    def _validate_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[0] > self.config.max_batch_size:
            raise ValueError("batch exceeds max_batch_size")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        if input_ids.shape[1] == 0:
            raise ValueError("input sequence must not be empty")
        return input_ids.to(device=self.device, dtype=torch.long)

    @torch.inference_mode()
    def prefill(
        self, input_ids: torch.Tensor, batch_max_length: int | None = None
    ) -> PrefillOutput:
        """
        处理完整 prompt，并把每层 K/V 写入一个全新的 cache。

        ``input_ids`` 的形状为 ``[batch, prompt_length]``。返回的 logits
        只保留每个请求最后一个有效 prompt token 对应的下一 token 分布。
        """
        input_ids = self._validate_input_ids(input_ids)
        batch_size, query_length = input_ids.shape
        pad_token_id = self.model.config.pad_token_id
        sequence_lengths = (input_ids != pad_token_id).sum(dim=1, dtype=torch.long)
        positions = torch.arange(query_length, device=self.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        self.cache.reset(batch_size)
        self._batch_size = batch_size
        model_input = Batch(
            input_ids=input_ids,
            positions=positions,
            sequence_lengths=sequence_lengths,
            curr_max_seq_len=batch_max_length or query_length,
            mode="prefill",
        )
        with measure_operation(self.device, self.config.synchronize_metrics) as metrics:
            logits = self.model(model_input, self.cache)
        batch_indices = torch.arange(batch_size, device=self.device)
        next_logits = logits[batch_indices, sequence_lengths - 1]
        return PrefillOutput(
            next_logits,
            sequence_lengths.clone(),
            metrics.latency_s,
            metrics.peak_allocated_bytes,
            metrics.peak_reserved_bytes,
        )

    @torch.inference_mode()
    def decode_step(
        self,
        token_ids: torch.Tensor,
        batch_max_length: int,
        active: torch.Tensor | None = None,
    ) -> DecodeOutput:
        """
        为 batch 中仍活跃的请求各追加一个 token。

        ``active`` 为 false 的 slot 仍保留在静态 batch 中，但其序列长度不会
        增加，因此已经完成的请求不会继续扩展 KV cache。
        """
        if self._batch_size == 0:
            raise RuntimeError("prefill() must be called before decode_step()")
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.shape != (self._batch_size, 1):
            raise ValueError("decode token_ids must have shape [batch] or [batch, 1]")
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        if active is None:
            active = torch.ones(self._batch_size, dtype=torch.bool, device=self.device)
        else:
            active = active.to(device=self.device, dtype=torch.bool)
        if active.shape != (self._batch_size,):
            raise ValueError("active must have shape [batch]")

        current_lengths = self.cache.lengths[: self._batch_size].clone()
        next_lengths = current_lengths + active.long()
        positions = current_lengths[:, None]
        model_input = Batch(
            input_ids=token_ids,
            positions=positions,
            sequence_lengths=next_lengths,
            curr_max_seq_len=batch_max_length,
            mode="decode",
        )
        with measure_operation(self.device, self.config.synchronize_metrics) as metrics:
            logits = self.model(model_input, self.cache, logits_to_keep=1)
        return DecodeOutput(
            logits[:, -1],
            next_lengths.clone(),
            metrics.latency_s,
            metrics.peak_allocated_bytes,
            metrics.peak_reserved_bytes,
        )

    def _encode_requests(
        self, requests: list[GenerationRequest]
    ) -> tuple[torch.Tensor, list[list[int]]]:
        encoded: list[list[int]] = []
        for request in requests:
            if request.input_ids is not None:
                encoded.append(list(request.input_ids))
            else:
                if self.tokenizer is None:
                    raise RuntimeError("a tokenizer is required for text prompts")
                encoded.append(
                    list(
                        self.tokenizer(request.prompt, add_special_tokens=True)[
                            "input_ids"
                        ]
                    )
                )
        for request, tokens in zip(requests, encoded, strict=True):
            if len(tokens) + request.max_new_tokens > self.config.max_sequence_length:
                raise ValueError(
                    "prompt and requested output exceed max_sequence_length"
                )
        max_length = max(map(len, encoded))
        if max_length > self.config.max_sequence_length:
            raise ValueError("prompt exceeds max_sequence_length")
        padded = torch.full(
            (len(encoded), max_length),
            self.model.config.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for row, tokens in enumerate(encoded):
            padded[row, : len(tokens)] = torch.tensor(tokens, device=self.device)
        return padded, encoded

    @torch.inference_mode()
    def generate(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        """对一个静态 batch 执行 prefill、采样和逐 token decode。"""
        if not requests:
            return []
        if len(requests) > self.config.max_batch_size:
            raise ValueError("request batch exceeds max_batch_size")

        # 阶段 1：tokenize 并右侧补齐，形成固定形状的 prompt batch。
        input_ids, encoded = self._encode_requests(requests)
        sampling_args = self.sampler.prepare(requests)
        scheduler = create_scheduler(
            self.config,
            default_stop_token_ids=(self.model.config.eos_token_id,),
        )
        states = []
        for request, prompt_token_ids in zip(requests, encoded, strict=True):
            state = RequestState(
                request=request,
                prompt_token_ids=prompt_token_ids,
            )
            scheduler.add_request(state)
            states.append(state)

        started = perf_counter()

        # 阶段 2：一次 prefill 处理全部 prompt，并产生第一个输出 token。
        prefill_schedule = scheduler.schedule()
        if prefill_schedule.mode != "prefill":
            raise RuntimeError("scheduler must begin with a prefill schedule")
        prefill_output = self.prefill(
            input_ids, batch_max_length=prefill_schedule.batch_max_length
        )
        logits = prefill_output.logits

        decode_latencies: list[list[float]] = [[] for _ in requests]
        peak_allocated_bytes = prefill_output.peak_allocated_bytes
        peak_reserved_bytes = prefill_output.peak_reserved_bytes

        if scheduler.has_unfinished_requests():
            next_tokens = self.sampler.sample(logits, sampling_args)
            scheduler.update(next_tokens.tolist())

        # 阶段 3：保持 batch slot 不变，每轮只为活跃请求 decode 一个 token。
        while scheduler.has_unfinished_requests():
            decode_schedule = scheduler.schedule()
            if decode_schedule.mode != "decode":
                raise RuntimeError("scheduler returned prefill after decoding started")

            active = torch.tensor(
                [
                    scheduled.num_scheduled_tokens > 0
                    for scheduled in decode_schedule.requests
                ],
                dtype=torch.bool,
                device=self.device,
            )
            token_ids = torch.tensor(
                [
                    (
                        scheduled.request.output_token_ids[-1]
                        if scheduled.num_scheduled_tokens > 0
                        else self.model.config.pad_token_id
                    )
                    for scheduled in decode_schedule.requests
                ],
                dtype=torch.long,
                device=self.device,
            )

            decode = self.decode_step(
                token_ids,
                batch_max_length=decode_schedule.batch_max_length,
                active=active,
            )
            logits = decode.logits

            peak_allocated_bytes = max(
                peak_allocated_bytes, decode.peak_allocated_bytes
            )
            peak_reserved_bytes = max(peak_reserved_bytes, decode.peak_reserved_bytes)
            for index in active.nonzero().flatten().tolist():
                decode_latencies[index].append(decode.latency_s)

            next_tokens = self.sampler.sample(logits, sampling_args)
            scheduler.update(next_tokens.tolist())

        # 阶段 4：将共享的 batch 执行结果整理为逐请求输出和指标。
        total_latency = perf_counter() - started
        outputs = []
        for index, state in enumerate(states):
            if self.tokenizer is None:
                text = ""
            else:
                text = self.tokenizer.decode(
                    state.output_token_ids, skip_special_tokens=True
                )
            outputs.append(
                GenerationOutput(
                    token_ids=state.output_token_ids,
                    text=text,
                    prompt_tokens=len(state.prompt_token_ids),
                    generated_tokens=len(state.output_token_ids),
                    finish_reason=state.finish_reason,
                    metrics=RequestMetrics(
                        prefill_latency_s=prefill_output.latency_s,
                        decode_latencies_s=tuple(decode_latencies[index]),
                        total_latency_s=total_latency,
                        peak_allocated_bytes=peak_allocated_bytes,
                        peak_reserved_bytes=peak_reserved_bytes,
                    ),
                )
            )
        return outputs

    def _validate_loss_mask(
        self,
        loss_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if loss_mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
        if loss_mask.shape != input_ids.shape:
            raise ValueError("loss_mask must have the same shape as input_ids")
        return loss_mask.to(device=self.device, dtype=torch.bool)

    @staticmethod
    def _make_score_output(
        nll_sum: torch.Tensor,
        token_count: int,
        *,
        logits: torch.Tensor | None = None,
        token_logprobs: torch.Tensor | None = None,
    ) -> ScoreOutput:
        if token_count <= 0:
            raise ValueError("score requires at least one selected target token")
        nll_sum_value = float(nll_sum.double().item())
        mean_nll = nll_sum_value / token_count
        return ScoreOutput(
            logits=logits,
            token_logprobs=token_logprobs,
            nll_sum=nll_sum_value,
            token_count=token_count,
            mean_nll=mean_nll,
            mean_logprob=-mean_nll,
            perplexity=math.exp(mean_nll),
        )

    @torch.inference_mode()
    def score(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> ScoreOutput:
        input_ids = self._validate_input_ids(input_ids)
        if input_ids.shape[1] < 2:
            raise ValueError("score requires at least two tokens")
        loss_mask = self._validate_loss_mask(loss_mask, input_ids)
        if chunk_size is not None:
            return self._score_chunked(input_ids, loss_mask, chunk_size)

        logits = self.model(input_ids)
        token_logprobs = (
            F.log_softmax(logits[:, :-1].float(), dim=-1)
            .gather(-1, input_ids[:, 1:, None])
            .squeeze(-1)
        )
        selected = token_logprobs[loss_mask[:, 1:]]
        return self._make_score_output(
            -selected.double().sum(),
            selected.numel(),
            logits=logits,
            token_logprobs=token_logprobs,
        )

    def _score_chunked(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        chunk_size: int,
    ) -> ScoreOutput:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        batch_size, sequence_length = input_ids.shape
        source_ids = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]
        target_mask = loss_mask[:, 1:]
        self.cache.reset(batch_size)
        self._batch_size = batch_size
        nll_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        token_count = 0

        for start in range(0, sequence_length - 1, chunk_size):
            end = min(start + chunk_size, sequence_length - 1)
            positions = torch.arange(start, end, device=self.device).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
            sequence_lengths = torch.full(
                (batch_size,),
                end,
                dtype=torch.long,
                device=self.device,
            )
            model_input = Batch(
                input_ids=source_ids[:, start:end],
                positions=positions,
                sequence_lengths=sequence_lengths,
                curr_max_seq_len=end,
                mode="prefill" if start == 0 else "decode",
            )
            logits = self.model(model_input, self.cache)
            chunk_logprobs = (
                F.log_softmax(logits.float(), dim=-1)
                .gather(-1, target_ids[:, start:end, None])
                .squeeze(-1)
            )
            selected = chunk_logprobs[target_mask[:, start:end]]
            nll_sum += -selected.double().sum()
            token_count += selected.numel()

        return self._make_score_output(nll_sum, token_count)
