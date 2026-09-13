"""Deferred host materialization for transactional sequence-policy traces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyTrace,
    NativePolicyTraceHostTransfer,
)
from ptcg_rl.rl.stateless_actor import (
    StatelessActorBatchTrace,
    StatelessActorDecisionTrace,
)

_FinishCallback = Callable[[], StatelessActorBatchTrace]
_TraceFinalizer = Callable[[NativePolicyNumpyTrace], StatelessActorBatchTrace]
_CancelCallback = Callable[[], None]


@dataclass(slots=True)
class SequenceActorHostTransfer:
    """Finalize one sequence actor host boundary exactly once."""

    _finish_callback: _FinishCallback | None = None
    _finish_ready_callback: _FinishCallback | None = None
    _cancel_callback: _CancelCallback | None = None
    _result: StatelessActorBatchTrace | None = None
    _native_trace: NativePolicyNumpyTrace | None = None
    _error: BaseException | None = None
    _finalizing: bool = False
    _cancelled: bool = False

    @classmethod
    def from_native_trace(
        cls,
        transfer: NativePolicyTraceHostTransfer,
        *,
        finalize: _TraceFinalizer,
        cancel: _CancelCallback | None = None,
    ) -> SequenceActorHostTransfer:
        """Wrap one packed native trace transfer and its actor finalizer."""

        def cancel_native() -> None:
            try:
                if transfer.ready_event is not None:
                    transfer.ready_event.synchronize()
            finally:
                if cancel is not None:
                    cancel()

        result = cls(_cancel_callback=cancel_native)

        def finish_native(*, ready: bool) -> StatelessActorBatchTrace:
            native_trace = transfer.finish_ready() if ready else transfer.finish()
            result._native_trace = native_trace
            return finalize(native_trace)

        result._finish_callback = lambda: finish_native(ready=False)
        result._finish_ready_callback = lambda: finish_native(ready=True)
        return result

    @classmethod
    def from_callbacks(
        cls,
        *,
        finish: _FinishCallback,
        finish_ready: _FinishCallback,
        cancel: _CancelCallback | None = None,
    ) -> SequenceActorHostTransfer:
        """Wrap dependent transfers such as an in-flight duplicate view."""
        return cls(
            _finish_callback=finish,
            _finish_ready_callback=finish_ready,
            _cancel_callback=cancel,
        )

    @classmethod
    def completed(
        cls,
        result: StatelessActorBatchTrace,
    ) -> SequenceActorHostTransfer:
        """Return an idempotent transfer for an already staged duplicate."""
        return cls(_result=result)

    def finish(self) -> StatelessActorBatchTrace:
        """Wait for the copy when necessary and finalize actor objects."""
        return self._resolve(ready=False)

    def finish_ready(self) -> StatelessActorBatchTrace:
        """Finalize after the owner has synchronized the copy stream."""
        return self._resolve(ready=True)

    def finish_native_ready(
        self,
    ) -> tuple[StatelessActorBatchTrace, NativePolicyNumpyTrace | None]:
        """Finalize and retain the compact source trace when one is available."""
        result = self.finish_ready()
        return result, self._native_trace

    def finish_native(
        self,
    ) -> tuple[StatelessActorBatchTrace, NativePolicyNumpyTrace | None]:
        """Wait, finalize, and retain the compact source trace when available."""
        result = self.finish()
        return result, self._native_trace

    def cancel(self) -> None:
        """Discard an unfinished transfer without staging temporal state."""
        if (
            self._result is not None
            or self._error is not None
            or self._cancelled
        ):
            return
        if self._finalizing:
            raise RuntimeError("sequence actor transfer is already finalizing")
        callback = self._cancel_callback
        self._cancelled = True
        try:
            if callback is not None:
                callback()
        finally:
            self._finish_callback = None
            self._finish_ready_callback = None
            self._cancel_callback = None

    def _resolve(self, *, ready: bool) -> StatelessActorBatchTrace:
        """Run one selected callback at most once, caching result or failure."""
        if self._result is not None:
            return self._result
        if self._error is not None:
            raise self._error
        if self._cancelled:
            raise RuntimeError("sequence actor transfer was cancelled")
        if self._finalizing:
            raise RuntimeError("sequence actor transfer is already finalizing")
        callback = (
            self._finish_ready_callback if ready else self._finish_callback
        )
        if callback is None:
            raise RuntimeError("sequence actor transfer has no finalizer")
        self._finalizing = True
        try:
            self._result = callback()
        except BaseException as error:
            self._error = error
            raise
        finally:
            self._finalizing = False
            self._finish_callback = None
            self._finish_ready_callback = None
            self._cancel_callback = None
        return self._result


def decisions_from_native_trace(
    trace: NativePolicyNumpyTrace,
) -> tuple[StatelessActorDecisionTrace, ...]:
    """Materialize the existing object trace contract from compact CSR arrays."""
    decisions: list[StatelessActorDecisionTrace] = []
    for row in range(trace.batch_size):
        action_start = int(trace.action_offsets[row])
        action_stop = int(trace.action_offsets[row + 1])
        token_start = int(trace.token_offsets[row])
        token_stop = int(trace.token_offsets[row + 1])
        decisions.append(
            StatelessActorDecisionTrace(
                action=tuple(
                    int(choice)
                    for choice in trace.action_choices[action_start:action_stop]
                ),
                action_logprob=float(trace.action_logprobs[row]),
                token_logprobs=tuple(
                    float(value)
                    for value in trace.token_logprobs[token_start:token_stop]
                ),
                prefix_values=tuple(
                    float(value)
                    for value in trace.prefix_values[token_start:token_stop]
                ),
                root_value=float(trace.root_values[row]),
                stop_sampled=bool(trace.stop_sampled[row]),
            )
        )
    return tuple(decisions)


__all__ = [
    "SequenceActorHostTransfer",
    "decisions_from_native_trace",
]
