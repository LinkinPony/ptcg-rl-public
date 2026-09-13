"""Device and compact host traces for tensor-native policy inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity

Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True, slots=True)
class NativePolicyTensorActionBatch:
    """Device-resident sampled actions without PPO behavior evidence."""

    identity: StatelessFragmentIdentity
    action_choices: Tensor
    action_lengths: Tensor

    def __post_init__(self) -> None:
        """Validate tensor structure without synchronizing device values."""
        _validate_tensor_actions(
            self.action_choices,
            self.action_lengths,
        )

    @property
    def batch_size(self) -> int:
        """Return the number of sampled decisions."""
        return int(self.action_lengths.shape[0])

    def to_host(self) -> NativePolicyNumpyActionBatch:
        """Compact sampled actions to host CSR arrays."""
        return _compact_actions_to_host(self)

    def defer_to_host(
        self,
        *,
        copy_stream: Any | None = None,
    ) -> NativePolicyActionHostTransfer:
        """Queue one packed D2H copy without synchronizing its producer."""
        return _defer_actions_to_host(self, copy_stream=copy_stream)


@dataclass(frozen=True, slots=True)
class NativePolicyTensorTrace:
    """Device-resident behavior evidence from one current-policy forward."""

    identity: StatelessFragmentIdentity
    action_choices: Tensor
    action_lengths: Tensor
    action_logprobs: Tensor
    token_logprobs: Tensor
    token_mask: Tensor
    prefix_values: Tensor
    root_values: Tensor
    stop_sampled: Tensor

    def __post_init__(self) -> None:
        """Validate tensor structure without synchronizing device values."""
        if self.action_choices.ndim != 2:
            raise ValueError("action choices must have shape [batch, choices]")
        batch_size = int(self.action_choices.shape[0])
        if batch_size <= 0:
            raise ValueError("native policy trace must contain decisions")
        if self.action_lengths.shape != (batch_size,):
            raise ValueError("action lengths must align with decisions")
        if self.action_logprobs.shape != (batch_size,):
            raise ValueError("action log-probs must align with decisions")
        token_shape = self.token_logprobs.shape
        if (
            len(token_shape) != 2
            or int(token_shape[0]) != batch_size
            or self.token_mask.shape != token_shape
            or self.prefix_values.shape != token_shape
        ):
            raise ValueError("decode-token tensors must share [batch, tokens]")
        if self.root_values.shape != (batch_size,):
            raise ValueError("root values must align with decisions")
        if self.stop_sampled.shape != (batch_size,):
            raise ValueError("STOP flags must align with decisions")
        if self.action_choices.dtype not in _INTEGER_DTYPES:
            raise TypeError("action choices must use an integer dtype")
        if self.action_lengths.dtype not in _INTEGER_DTYPES:
            raise TypeError("action lengths must use an integer dtype")
        if self.token_mask.dtype != torch.bool:
            raise TypeError("decode-token mask must use bool dtype")
        if self.stop_sampled.dtype != torch.bool:
            raise TypeError("STOP flags must use bool dtype")
        devices = {
            tensor.device
            for tensor in (
                self.action_choices,
                self.action_lengths,
                self.action_logprobs,
                self.token_logprobs,
                self.token_mask,
                self.prefix_values,
                self.root_values,
                self.stop_sampled,
            )
        }
        if len(devices) != 1:
            raise ValueError("native policy trace tensors must share one device")

    @property
    def batch_size(self) -> int:
        """Return the number of sampled decisions."""
        return int(self.action_lengths.shape[0])

    def to_host(self) -> NativePolicyNumpyTrace:
        """Compact this trace to host CSR arrays with one bulk transfer."""
        return _compact_trace_to_host(self)

    def defer_to_host(
        self,
        *,
        copy_stream: Any | None = None,
    ) -> NativePolicyTraceHostTransfer:
        """Queue one packed D2H copy without synchronizing its producer."""
        return _defer_trace_to_host(self, copy_stream=copy_stream)


def evaluation_action_only_tensor_trace(
    *,
    identity: StatelessFragmentIdentity,
    action_choices: Tensor,
    action_lengths: Tensor,
    stop_sampled: Tensor,
) -> NativePolicyTensorTrace:
    """Build minimal compatibility storage for a trajectory-free evaluator.

    Native arenas currently scatter current-policy actions through the same
    transactional object used by training.  Evaluation-only greedy collection
    still needs that transaction and STOP flag, but does not need root values,
    prefix values, log-probabilities, or random samples.  The one-token zeros
    below are structural placeholders only.  Callers must prove that trajectory
    retention is disabled before requesting this path.
    """
    _validate_tensor_actions(action_choices, action_lengths)
    batch_size = int(action_lengths.shape[0])
    if stop_sampled.shape != (batch_size,) or stop_sampled.dtype != torch.bool:
        raise ValueError("evaluation STOP flags must align with actions")
    if stop_sampled.device != action_lengths.device:
        raise ValueError("evaluation actions and STOP flags must share one device")
    evidence = torch.zeros(
        (batch_size, 1),
        dtype=torch.float32,
        device=action_lengths.device,
    )
    return NativePolicyTensorTrace(
        identity=identity,
        action_choices=action_choices,
        action_lengths=action_lengths,
        action_logprobs=evidence[:, 0],
        token_logprobs=evidence,
        token_mask=torch.ones_like(evidence, dtype=torch.bool),
        prefix_values=evidence,
        root_values=evidence[:, 0],
        stop_sampled=stop_sampled,
    )


@dataclass(frozen=True, slots=True)
class NativePolicyNumpyActionBatch:
    """Compact host sampled actions for an opponent engine step."""

    identity: StatelessFragmentIdentity
    action_offsets: Int64Array
    action_choices: Int32Array

    def __post_init__(self) -> None:
        """Reject malformed actions before passing them to the engine."""
        _validate_offsets(
            self.action_offsets,
            value_count=int(self.action_choices.size),
            name="action",
        )
        if int(self.action_offsets.size) <= 1:
            raise ValueError("native policy actions must contain decisions")
        if self.action_offsets.dtype != np.int64:
            raise TypeError("action offsets must use int64")
        if self.action_choices.dtype != np.int32:
            raise TypeError("action choices must use int32")
        if np.any(self.action_choices < 0):
            raise ValueError("sampled action choices must be non-negative")
        self.action_offsets.setflags(write=False)
        self.action_choices.setflags(write=False)

    @property
    def batch_size(self) -> int:
        """Return the number of sampled decisions."""
        return int(self.action_offsets.size) - 1


@dataclass(frozen=True, slots=True)
class NativePolicyNumpyTrace:
    """Compact host behavior evidence ready for an array trajectory page."""

    identity: StatelessFragmentIdentity
    action_offsets: Int64Array
    action_choices: Int32Array
    action_logprobs: Float32Array
    token_offsets: Int64Array
    token_logprobs: Float32Array
    prefix_values: Float32Array
    root_values: Float32Array
    stop_sampled: BoolArray

    def __post_init__(self) -> None:
        """Reject malformed evidence before it can enter a trajectory page."""
        _validate_offsets(
            self.action_offsets,
            value_count=int(self.action_choices.size),
            name="action",
        )
        _validate_offsets(
            self.token_offsets,
            value_count=int(self.token_logprobs.size),
            name="decode-token",
        )
        batch_size = int(self.action_offsets.size) - 1
        if batch_size <= 0:
            raise ValueError("native policy trace must contain decisions")
        if self.token_offsets.shape != (batch_size + 1,):
            raise ValueError("action and decode-token CSR rows must align")
        for name, values in (
            ("action_logprobs", self.action_logprobs),
            ("root_values", self.root_values),
            ("stop_sampled", self.stop_sampled),
        ):
            if values.shape != (batch_size,):
                raise ValueError(f"{name} must align with decisions")
        if self.prefix_values.shape != self.token_logprobs.shape:
            raise ValueError("prefix values must align with decode-token log-probs")
        if self.action_offsets.dtype != np.int64:
            raise TypeError("action offsets must use int64")
        if self.action_choices.dtype != np.int32:
            raise TypeError("action choices must use int32")
        if self.token_offsets.dtype != np.int64:
            raise TypeError("decode-token offsets must use int64")
        for name, values in (
            ("action_logprobs", self.action_logprobs),
            ("token_logprobs", self.token_logprobs),
            ("prefix_values", self.prefix_values),
            ("root_values", self.root_values),
        ):
            if values.dtype != np.float32:
                raise TypeError(f"{name} must use float32")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite")
        if self.stop_sampled.dtype != np.bool_:
            raise TypeError("STOP flags must use bool")
        if np.any(np.diff(self.token_offsets) <= 0):
            raise ValueError("each decision requires behavior decode-token evidence")
        if np.any(self.action_choices < 0):
            raise ValueError("sampled action choices must be non-negative")
        token_prefix: npt.NDArray[np.float64] = np.zeros(
            self.token_logprobs.size + 1,
            dtype=np.float64,
        )
        token_prefix[1:] = np.cumsum(
            self.token_logprobs,
            dtype=np.float64,
        )
        token_sums = (
            token_prefix[self.token_offsets[1:]] - token_prefix[self.token_offsets[:-1]]
        )
        if not np.allclose(
            self.action_logprobs,
            token_sums,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("action log-probs differ from decode-token sums")
        for array in (
            self.action_offsets,
            self.action_choices,
            self.action_logprobs,
            self.token_offsets,
            self.token_logprobs,
            self.prefix_values,
            self.root_values,
            self.stop_sampled,
        ):
            array.setflags(write=False)

    @property
    def batch_size(self) -> int:
        """Return the number of sampled decisions."""
        return int(self.action_offsets.size) - 1

    @property
    def behavior_policy_version(self) -> int:
        """Return the exact behavior publication version."""
        return self.identity.behavior_policy_version

    @property
    def behavior_policy_fingerprint(self) -> str:
        """Return the exact behavior model-state fingerprint."""
        return self.identity.behavior_policy_fingerprint

    @property
    def input_contract_fingerprint(self) -> str:
        """Return the immutable behavior input contract."""
        return self.identity.input_contract_fingerprint


@dataclass(slots=True)
class NativePolicyActionHostTransfer:
    """One queued action-only D2H transfer retaining its exact identity."""

    identity: StatelessFragmentIdentity
    host_bits: Tensor
    device_bits: Tensor
    ready_event: Any | None
    choice_width: int
    _result: NativePolicyNumpyActionBatch | None = None

    def finish(self) -> NativePolicyNumpyActionBatch:
        """Wait only when needed, then materialize immutable host CSR."""
        if self.ready_event is not None:
            self.ready_event.synchronize()
        return self.finish_ready()

    def finish_ready(self) -> NativePolicyNumpyActionBatch:
        """Materialize host CSR after the owning bank synchronized its stream."""
        if self._result is None:
            self._result = _actions_from_host_bits(
                _host_numpy(self.host_bits),
                identity=self.identity,
                choice_width=self.choice_width,
            )
            self.device_bits = torch.empty(0, dtype=torch.int32)
        return self._result


@dataclass(slots=True)
class NativePolicyTraceHostTransfer:
    """One queued full-trace D2H transfer retaining its behavior identity."""

    identity: StatelessFragmentIdentity
    host_bits: Tensor
    device_bits: Tensor
    ready_event: Any | None
    choice_width: int
    token_width: int
    _result: NativePolicyNumpyTrace | None = None

    def finish(self) -> NativePolicyNumpyTrace:
        """Wait only when needed, then materialize immutable host CSR."""
        if self.ready_event is not None:
            self.ready_event.synchronize()
        return self.finish_ready()

    def finish_ready(self) -> NativePolicyNumpyTrace:
        """Materialize host CSR after the owning bank synchronized its stream."""
        if self._result is None:
            self._result = _trace_from_host_bits(
                _host_numpy(self.host_bits),
                identity=self.identity,
                choice_width=self.choice_width,
                token_width=self.token_width,
            )
            self.device_bits = torch.empty(0, dtype=torch.int32)
        return self._result


def _compact_actions_to_host(
    actions: NativePolicyTensorActionBatch,
) -> NativePolicyNumpyActionBatch:
    """Pack action tensors once, synchronize once, and form host CSR."""
    return _actions_from_host_bits(
        np.asarray(
            _pack_action_bits(actions).cpu().numpy(),
            dtype=np.int32,
        ),
        identity=actions.identity,
        choice_width=int(actions.action_choices.shape[1]),
    )


def _defer_actions_to_host(
    actions: NativePolicyTensorActionBatch,
    *,
    copy_stream: Any | None,
) -> NativePolicyActionHostTransfer:
    """Pack action tensors and enqueue exactly one device-to-host copy."""
    device_bits = _pack_action_bits(actions)
    host_bits, ready_event = _defer_int32_bits_to_host(
        device_bits,
        copy_stream=copy_stream,
    )
    return NativePolicyActionHostTransfer(
        identity=actions.identity,
        host_bits=host_bits,
        device_bits=device_bits,
        ready_event=ready_event,
        choice_width=int(actions.action_choices.shape[1]),
    )


def _pack_action_bits(actions: NativePolicyTensorActionBatch) -> Tensor:
    """Pack padded choices and lengths into one int32 device tensor."""
    return torch.cat(
        (
            actions.action_choices.detach().to(dtype=torch.int32),
            actions.action_lengths.detach().to(dtype=torch.int32).unsqueeze(1),
        ),
        dim=1,
    )


def _actions_from_host_bits(
    host_bits: Int32Array,
    *,
    identity: StatelessFragmentIdentity,
    choice_width: int,
) -> NativePolicyNumpyActionBatch:
    """Form one compact action batch from synchronized packed host bits."""
    action_lengths: Int64Array = host_bits[:, choice_width].astype(
        np.int64,
        copy=False,
    )
    if np.any(action_lengths < 0) or np.any(action_lengths > choice_width):
        raise ValueError("sampled action lengths exceed padded action width")
    action_mask = (
        np.arange(choice_width, dtype=np.int64)[None, :] < action_lengths[:, None]
    )
    return NativePolicyNumpyActionBatch(
        identity=identity,
        action_offsets=_offsets(action_lengths),
        action_choices=np.asarray(
            host_bits[:, :choice_width][action_mask],
            dtype=np.int32,
        ),
    )


def _compact_trace_to_host(
    trace: NativePolicyTensorTrace,
) -> NativePolicyNumpyTrace:
    """Bit-pack mixed trace dtypes, synchronize once, then form host CSR."""
    return _trace_from_host_bits(
        np.asarray(
            _pack_trace_bits(trace).cpu().numpy(),
            dtype=np.int32,
        ),
        identity=trace.identity,
        choice_width=int(trace.action_choices.shape[1]),
        token_width=int(trace.token_logprobs.shape[1]),
    )


def _defer_trace_to_host(
    trace: NativePolicyTensorTrace,
    *,
    copy_stream: Any | None,
) -> NativePolicyTraceHostTransfer:
    """Bit-pack full behavior evidence and queue one D2H copy."""
    device_bits = _pack_trace_bits(trace)
    host_bits, ready_event = _defer_int32_bits_to_host(
        device_bits,
        copy_stream=copy_stream,
    )
    return NativePolicyTraceHostTransfer(
        identity=trace.identity,
        host_bits=host_bits,
        device_bits=device_bits,
        ready_event=ready_event,
        choice_width=int(trace.action_choices.shape[1]),
        token_width=int(trace.token_logprobs.shape[1]),
    )


def _pack_trace_bits(trace: NativePolicyTensorTrace) -> Tensor:
    """Pack every behavior-evidence dtype into one int32 device tensor."""
    return torch.cat(
        (
            trace.action_choices.detach().to(dtype=torch.int32),
            trace.action_lengths.detach().to(dtype=torch.int32).unsqueeze(1),
            _float32_bits(trace.action_logprobs).unsqueeze(1),
            _float32_bits(trace.token_logprobs),
            trace.token_mask.detach().to(dtype=torch.int32),
            _float32_bits(trace.prefix_values),
            _float32_bits(trace.root_values).unsqueeze(1),
            trace.stop_sampled.detach().to(dtype=torch.int32).unsqueeze(1),
        ),
        dim=1,
    )


def _defer_int32_bits_to_host(
    device_bits: Tensor,
    *,
    copy_stream: Any | None,
) -> tuple[Tensor, Any | None]:
    """Queue packed bits on pinned host memory and return a completion event."""
    if device_bits.dtype != torch.int32:
        raise TypeError("native policy host transfer requires packed int32 bits")
    if device_bits.device.type != "cuda":
        if copy_stream is not None:
            raise ValueError("CPU policy transfer cannot use a CUDA copy stream")
        return device_bits.detach().cpu(), None
    producer_stream = torch.cuda.current_stream(device_bits.device)
    resolved_copy_stream = producer_stream if copy_stream is None else copy_stream
    event_factory = cast(Any, torch.cuda.Event)
    producer_ready = event_factory()
    producer_ready.record(producer_stream)
    host_bits = torch.empty(
        device_bits.shape,
        dtype=torch.int32,
        device="cpu",
        pin_memory=True,
    )
    with torch.cuda.stream(resolved_copy_stream):
        resolved_copy_stream.wait_event(producer_ready)
        host_bits.copy_(device_bits, non_blocking=True)
        ready_event = event_factory()
        ready_event.record(resolved_copy_stream)
    device_bits.record_stream(resolved_copy_stream)
    return host_bits, ready_event


def _host_numpy(host_bits: Tensor) -> Int32Array:
    """Expose synchronized packed host storage as a typed NumPy array."""
    if host_bits.device.type != "cpu" or host_bits.dtype != torch.int32:
        raise TypeError("native policy transfer did not produce CPU int32 storage")
    return np.asarray(host_bits.numpy(), dtype=np.int32)


def _trace_from_host_bits(
    host_bits: Int32Array,
    *,
    identity: StatelessFragmentIdentity,
    choice_width: int,
    token_width: int,
) -> NativePolicyNumpyTrace:
    """Form compact immutable behavior evidence from transferred int32 bits."""

    length_column = choice_width
    action_logprob_column = length_column + 1
    token_logprob_start = action_logprob_column + 1
    token_mask_start = token_logprob_start + token_width
    prefix_start = token_mask_start + token_width
    root_column = prefix_start + token_width
    stop_column = root_column + 1

    action_lengths: Int64Array = host_bits[:, length_column].astype(
        np.int64,
        copy=False,
    )
    if np.any(action_lengths < 0) or np.any(action_lengths > choice_width):
        raise ValueError("sampled action lengths exceed padded action width")
    action_offsets = _offsets(action_lengths)
    action_mask = (
        np.arange(choice_width, dtype=np.int64)[None, :] < action_lengths[:, None]
    )
    action_choices = np.asarray(
        host_bits[:, :choice_width][action_mask],
        dtype=np.int32,
    )
    token_mask: BoolArray = host_bits[
        :,
        token_mask_start : token_mask_start + token_width,
    ].astype(np.bool_, copy=False)
    token_offsets = _offsets(token_mask.sum(axis=1, dtype=np.int64))
    return NativePolicyNumpyTrace(
        identity=identity,
        action_offsets=action_offsets,
        action_choices=action_choices,
        action_logprobs=_float32_columns(
            host_bits,
            action_logprob_column,
            1,
        ).reshape(-1),
        token_offsets=token_offsets,
        token_logprobs=_float32_columns(
            host_bits,
            token_logprob_start,
            token_width,
        )[token_mask],
        prefix_values=_float32_columns(
            host_bits,
            prefix_start,
            token_width,
        )[token_mask],
        root_values=_float32_columns(
            host_bits,
            root_column,
            1,
        ).reshape(-1),
        stop_sampled=host_bits[:, stop_column].astype(np.bool_, copy=True),
    )


def _validate_tensor_actions(
    action_choices: Tensor,
    action_lengths: Tensor,
) -> None:
    """Validate padded device action tensors without reading their values."""
    if action_choices.ndim != 2:
        raise ValueError("action choices must have shape [batch, choices]")
    batch_size = int(action_choices.shape[0])
    if batch_size <= 0:
        raise ValueError("native policy actions must contain decisions")
    if action_lengths.shape != (batch_size,):
        raise ValueError("action lengths must align with decisions")
    if action_choices.dtype not in _INTEGER_DTYPES:
        raise TypeError("action choices must use an integer dtype")
    if action_lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError("action lengths must use an integer dtype")
    if action_choices.device != action_lengths.device:
        raise ValueError("native policy action tensors must share one device")


def _float32_bits(values: Tensor) -> Tensor:
    """Return exact int32 bit patterns for one float32 tensor."""
    return values.detach().to(dtype=torch.float32).contiguous().view(dtype=torch.int32)


def _float32_columns(
    values: Int32Array,
    start: int,
    width: int,
) -> Float32Array:
    """Recover copied float32 columns from their exact int32 bit patterns."""
    bits = np.ascontiguousarray(values[:, start : start + width])
    return cast(Float32Array, bits.view(np.float32))


def _offsets(lengths: Int64Array) -> Int64Array:
    """Return canonical int64 CSR offsets."""
    result: Int64Array = np.zeros(lengths.size + 1, dtype=np.int64)
    result[1:] = np.cumsum(lengths, dtype=np.int64)
    return result


def _validate_offsets(
    offsets: Int64Array,
    *,
    value_count: int,
    name: str,
) -> None:
    """Validate one canonical CSR offset vector."""
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"{name} offsets must contain at least one row")
    if offsets.dtype != np.int64:
        raise TypeError(f"{name} offsets must use int64")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != value_count
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"{name} offsets are invalid")


__all__ = [
    "NativePolicyActionHostTransfer",
    "NativePolicyNumpyActionBatch",
    "NativePolicyNumpyTrace",
    "NativePolicyTraceHostTransfer",
    "NativePolicyTensorActionBatch",
    "NativePolicyTensorTrace",
    "evaluation_action_only_tensor_trace",
]
