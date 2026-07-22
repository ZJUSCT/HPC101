from hpc101_infer.scheduler.base import (
    RequestState,
    RequestStatus,
    ScheduledOutput,
    ScheduledRequest,
)


class StaticBatchScheduler:
    def __init__(self, default_stop_token_ids: tuple[int, ...]) -> None:
        self.default_stop_token_ids = default_stop_token_ids
        self.requests: list[RequestState] = []
        self._prefill_scheduled = False

    def add_request(self, request: RequestState) -> None:
        if self._prefill_scheduled:
            raise RuntimeError(
                "cannot add requests after static batch scheduling starts"
            )
        if request.status is not RequestStatus.PENDING:
            raise ValueError("new requests must be pending")
        self.requests.append(request)

    def schedule(self) -> ScheduledOutput:
        if not self.requests:
            raise RuntimeError("cannot schedule an empty batch")

        if not self._prefill_scheduled:
            self._prefill_scheduled = True
            scheduled = []
            for state in self.requests:
                state.num_computed_tokens = len(state.prompt_token_ids)
                if state.request.max_new_tokens == 0:
                    state.status = RequestStatus.COMPLETED
                    state.finish_reason = "length"
                else:
                    state.status = RequestStatus.PREFILLING
                scheduled.append(
                    ScheduledRequest(
                        request=state,
                        num_scheduled_tokens=len(state.prompt_token_ids),
                    )
                )
            return ScheduledOutput(mode="prefill", requests=scheduled)

        if not self.has_unfinished_requests():
            raise RuntimeError("cannot schedule a completed batch")

        scheduled = []
        for state in self.requests:
            num_scheduled_tokens = int(state.status is RequestStatus.DECODING)
            state.num_computed_tokens += num_scheduled_tokens
            scheduled.append(
                ScheduledRequest(
                    request=state,
                    num_scheduled_tokens=num_scheduled_tokens,
                )
            )
        return ScheduledOutput(mode="decode", requests=scheduled)

    def update(self, token_ids: list[int]) -> None:
        if len(token_ids) != len(self.requests):
            raise ValueError("token_ids must match the static batch size")

        for state, token_id in zip(self.requests, token_ids, strict=True):
            if state.status not in (
                RequestStatus.PREFILLING,
                RequestStatus.DECODING,
            ):
                continue

            state.output_token_ids.append(token_id)
            stop_token_ids = (
                self.default_stop_token_ids
                if state.request.stop_token_ids is None
                else state.request.stop_token_ids
            )
            if token_id in stop_token_ids:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "stop"
            elif len(state.output_token_ids) >= state.request.max_new_tokens:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "length"
            else:
                state.status = RequestStatus.DECODING

    def has_unfinished_requests(self) -> bool:
        return any(
            state.status not in (RequestStatus.COMPLETED, RequestStatus.FAILED)
            for state in self.requests
        )
