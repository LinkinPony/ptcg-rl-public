"""Tensor-native current-policy inference and compact behavior traces."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import groupby
from typing import Protocol, TypeAlias

import torch
from torch import Tensor

from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.simple_stateless import (
    PublicBeliefSummaryBatch,
    SimpleExactRoutePlan,
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
    resolve_simple_exact_routes,
)
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.model.tensor_validation import require_tensor_condition
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyActionBatch,
    NativePolicyNumpyTrace,
    NativePolicyTensorActionBatch,
    NativePolicyTensorTrace,
    evaluation_action_only_tensor_trace,
)
from ptcg_rl.rl.policy_inputs import SimpleStatelessPolicyInputBatch
from ptcg_rl.rl.stateless_actor import model_uses_pure_bfloat16
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity


class TensorNativePolicyInputBatch(Protocol):
    """Object-light model inputs accepted by current-policy inference."""

    @property
    def states(self) -> StateBatch: ...

    @property
    def options(self) -> OptionBatch: ...

    @property
    def unique_deck_card_ids(self) -> Tensor: ...

    @property
    def deck_counts(self) -> Tensor: ...

    @property
    def deck_valid_mask(self) -> Tensor: ...

    @property
    def deck_signatures(self) -> tuple[str, ...]: ...

    @property
    def belief_summary(self) -> PublicBeliefSummaryBatch: ...

    @property
    def min_counts(self) -> tuple[int, ...]: ...

    @property
    def max_counts(self) -> tuple[int, ...]: ...

    @property
    def input_contract_fingerprint(self) -> str: ...

    @property
    def batch_size(self) -> int:
        """Return the number of aligned decisions."""


PolicyInputBatch: TypeAlias = (
    SimpleStatelessPolicyInputBatch | TensorNativePolicyInputBatch
)


@dataclass(frozen=True, slots=True)
class _ExactDeckTokenCache:
    """Executor-local exact-deck inputs and their frozen model tokens."""

    signatures: tuple[str, ...]
    signature_rows: dict[str, int]
    card_ids: Tensor
    counts: Tensor
    valid_mask: Tensor
    tokens: Tensor


@dataclass(frozen=True, slots=True)
class _ResolvedExactDeckBatch:
    """One repeated signature layout resolved to routes and cache rows."""

    route_plan: SimpleExactRoutePlan
    cache_rows: Tensor


class NativePolicyInferenceExecutor:
    """Run current-policy inference without actor-row compatibility objects."""

    def __init__(
        self,
        model: SimpleStatelessPolicyValueNet,
        *,
        identity: StatelessFragmentIdentity,
        device: torch.device | str,
        verify_model_state: bool = True,
    ) -> None:
        """Bind one immutable behavior publication to one inference device."""
        self.device = torch.device(device)
        self.identity = identity
        self.model = model.to(self.device).eval()
        self._uses_pure_bfloat16 = model_uses_pure_bfloat16(self.model)
        self._verify_contract()
        if verify_model_state:
            actual = canonical_model_state_fingerprint(self.model)
            if actual != identity.behavior_policy_fingerprint:
                raise ValueError(
                    "native inference model state differs from behavior identity"
                )
        self._resolved_deck_batches: OrderedDict[
            tuple[str, ...],
            _ResolvedExactDeckBatch,
        ] = OrderedDict()
        self._exact_deck_cache = self._build_exact_deck_cache()

    @property
    def uses_bfloat16_autocast(self) -> bool:
        """Return whether inference uses the H200 BF16 compute path."""
        return self.device.type == "cuda" and not self._uses_pure_bfloat16

    @property
    def uses_pure_bfloat16(self) -> bool:
        """Return whether all resident floating model state already uses BF16."""
        return self._uses_pure_bfloat16

    def sample_device(
        self,
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
        evaluation_action_only: bool = False,
    ) -> NativePolicyTensorTrace:
        """Sample one model-ready batch and retain all evidence on its device."""
        self._validate_batch(batch)
        self._validate_generator(generator)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("sampling temperature must be finite and non-negative")
        if evaluation_action_only != (temperature == 0.0):
            raise ValueError(
                "T=0 native inference requires evaluation_action_only and "
                "evaluation_action_only requires T=0"
            )
        if evaluation_action_only and sampling_uniforms is not None:
            raise ValueError("greedy evaluation cannot consume sampling uniforms")
        if not evaluation_action_only and temperature <= 0.0:
            raise ValueError("sampling temperature must be finite and positive")
        resolved = self._resolve_exact_deck_batch(batch.deck_signatures)
        self._validate_exact_deck_inputs(batch, resolved.cache_rows)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.uses_bfloat16_autocast,
            ),
        ):
            deck_tokens = self._exact_deck_cache.tokens.index_select(
                0,
                resolved.cache_rows,
            )
            state = self.model.encode_observation_state_with_deck_tokens(
                state=batch.states,
                deck_tokens=deck_tokens,
                belief_summary=batch.belief_summary,
                route_plan=resolved.route_plan,
            )
            option_embeddings = self.model.encode_legal_options(
                state,
                batch.options,
                route_plan=resolved.route_plan,
            )
            if evaluation_action_only:
                greedy = self.model.heads.greedy_decode_trace(
                    state.policy,
                    state.opponent_belief,
                    option_embeddings,
                    batch.options,
                    route_plan=resolved.route_plan,
                )
                return evaluation_action_only_tensor_trace(
                    identity=self.identity,
                    action_choices=greedy.actions.choice_indices,
                    action_lengths=greedy.actions.lengths,
                    stop_sampled=greedy.stop_sampled,
                )
            sampled = self.model.heads.sample_decode_with_trace(
                state.policy,
                state.opponent_belief,
                option_embeddings,
                batch.options,
                route_plan=resolved.route_plan,
                temperature=temperature,
                generator=generator,
                sampling_uniforms=sampling_uniforms,
            )
            root_values = self.model.heads.root_value(
                state.value,
                state.opponent_belief,
                route_plan=resolved.route_plan,
            )
        return NativePolicyTensorTrace(
            identity=self.identity,
            action_choices=sampled.actions.choice_indices,
            action_lengths=sampled.actions.lengths,
            action_logprobs=sampled.action_logprobs.float(),
            token_logprobs=sampled.token_logprobs.float(),
            token_mask=sampled.token_mask,
            prefix_values=sampled.prefix_values.float(),
            root_values=root_values.float(),
            stop_sampled=sampled.stop_sampled,
        )

    def sample_actions_device(
        self,
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
    ) -> NativePolicyTensorActionBatch:
        """Sample engine actions without evaluating or retaining PPO evidence."""
        self._validate_batch(batch)
        self._validate_generator(generator)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("sampling temperature must be finite and non-negative")
        if temperature == 0.0 and sampling_uniforms is not None:
            raise ValueError("greedy inference cannot consume sampling uniforms")
        resolved = self._resolve_exact_deck_batch(batch.deck_signatures)
        self._validate_exact_deck_inputs(batch, resolved.cache_rows)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.uses_bfloat16_autocast,
            ),
        ):
            deck_tokens = self._exact_deck_cache.tokens.index_select(
                0,
                resolved.cache_rows,
            )
            state = self.model.encode_observation_state_with_deck_tokens(
                state=batch.states,
                deck_tokens=deck_tokens,
                belief_summary=batch.belief_summary,
                route_plan=resolved.route_plan,
            )
            option_embeddings = self.model.encode_legal_options(
                state,
                batch.options,
                route_plan=resolved.route_plan,
            )
            sampled = (
                self.model.heads.greedy_decode_actions(
                    state.policy,
                    state.opponent_belief,
                    option_embeddings,
                    batch.options,
                    route_plan=resolved.route_plan,
                )
                if temperature == 0.0
                else self.model.heads.sample_decode_actions(
                    state.policy,
                    state.opponent_belief,
                    option_embeddings,
                    batch.options,
                    route_plan=resolved.route_plan,
                    temperature=temperature,
                    generator=generator,
                    sampling_uniforms=sampling_uniforms,
                )
            )
        return NativePolicyTensorActionBatch(
            identity=self.identity,
            action_choices=sampled.choice_indices,
            action_lengths=sampled.lengths,
        )

    def sample(
        self,
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
        evaluation_action_only: bool = False,
    ) -> NativePolicyNumpyTrace:
        """Sample and return compact host arrays after one bulk D2H."""
        return self.sample_device(
            batch,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
            evaluation_action_only=evaluation_action_only,
        ).to_host()

    def sample_actions(
        self,
        batch: PolicyInputBatch,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_uniforms: Tensor | None = None,
    ) -> NativePolicyNumpyActionBatch:
        """Synchronously sample only engine actions."""
        return self.sample_actions_device(
            batch,
            temperature=temperature,
            generator=generator,
            sampling_uniforms=sampling_uniforms,
        ).to_host()

    def _resolve_exact_deck_batch(
        self,
        deck_signatures: Sequence[str],
    ) -> _ResolvedExactDeckBatch:
        """Reuse exact route and deck-cache rows for repeated batch layouts."""
        key = tuple(deck_signatures)
        cached = self._resolved_deck_batches.get(key)
        if cached is not None:
            self._resolved_deck_batches.move_to_end(key)
            return cached
        resolved = _ResolvedExactDeckBatch(
            route_plan=resolve_simple_exact_routes(
                key,
                self.model.config,
                device=self.device,
            ),
            cache_rows=torch.tensor(
                tuple(self._exact_deck_cache.signature_rows[item] for item in key),
                dtype=torch.long,
                device=self.device,
            ),
        )
        self._resolved_deck_batches[key] = resolved
        if len(self._resolved_deck_batches) > 256:
            self._resolved_deck_batches.popitem(last=False)
        return resolved

    def _build_exact_deck_cache(self) -> _ExactDeckTokenCache:
        """Preencode every immutable exact route once for this executor."""
        signatures, card_ids, counts, valid_mask = _collate_exact_routes(
            self.model.config,
        )
        card_ids = card_ids.to(device=self.device)
        counts = counts.to(device=self.device)
        valid_mask = valid_mask.to(device=self.device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.uses_bfloat16_autocast,
            ),
        ):
            tokens = self.model.encode_exact_deck_tokens(
                unique_deck_card_ids=card_ids,
                deck_counts=counts,
                deck_valid_mask=valid_mask,
            )
        expected_shape = (len(signatures), self.model.config.d_model)
        device_mismatch = tokens.device.type != self.device.type or (
            self.device.index is not None and tokens.device.index != self.device.index
        )
        if tuple(tokens.shape) != expected_shape or device_mismatch:
            raise ValueError(
                "preencoded exact-deck token table differs from model configuration"
            )
        return _ExactDeckTokenCache(
            signatures=signatures,
            signature_rows={signature: row for row, signature in enumerate(signatures)},
            card_ids=card_ids,
            counts=counts,
            valid_mask=valid_mask,
            tokens=tokens,
        )

    def _validate_exact_deck_inputs(
        self,
        batch: PolicyInputBatch,
        cache_rows: Tensor,
    ) -> None:
        """Require raw deck tensors to match their declared exact signatures."""
        shape = batch.unique_deck_card_ids.shape
        if len(shape) != 2:
            raise ValueError("native exact-deck card IDs must be rank two")
        width = int(shape[1])
        cache_width = int(self._exact_deck_cache.card_ids.shape[1])
        if width <= 0 or width > cache_width:
            raise ValueError("native exact-deck width differs from cached registry")
        expected_card_ids = self._exact_deck_cache.card_ids.index_select(
            0,
            cache_rows,
        )
        expected_counts = self._exact_deck_cache.counts.index_select(0, cache_rows)
        expected_valid = self._exact_deck_cache.valid_mask.index_select(0, cache_rows)
        condition = (
            expected_card_ids[:, :width].eq(batch.unique_deck_card_ids).all()
            & expected_counts[:, :width]
            .eq(batch.deck_counts.to(dtype=expected_counts.dtype))
            .all()
            & expected_valid[:, :width].eq(batch.deck_valid_mask).all()
            & (~expected_valid[:, width:]).all()
        )
        require_tensor_condition(
            condition,
            "native exact-deck tensors differ from their declared signatures",
        )

    def _validate_batch(self, batch: PolicyInputBatch) -> None:
        batch_size = int(batch.batch_size)
        if batch_size <= 0:
            raise ValueError("native policy inference requires at least one row")
        if batch.input_contract_fingerprint != self.identity.input_contract_fingerprint:
            raise ValueError("native policy batch differs from behavior input contract")
        catalog = batch.belief_summary.catalog_fingerprint
        declared_catalog = getattr(
            batch,
            "public_deck_catalog_fingerprint",
            catalog,
        )
        if declared_catalog != catalog:
            raise ValueError("native policy batch catalog metadata is inconsistent")
        if catalog != self.identity.public_deck_catalog_fingerprint:
            raise ValueError("native policy batch differs from behavior public catalog")
        if (
            len(batch.deck_signatures) != batch_size
            or len(batch.min_counts) != batch_size
            or len(batch.max_counts) != batch_size
        ):
            raise ValueError("native policy batch host rows are misaligned")
        if any(maximum <= 0 for maximum in batch.max_counts):
            raise ValueError("zero-card forced prompts must bypass policy sampling")
        option_width = int(batch.options.valid_options.shape[1])
        if any(
            minimum < 0 or minimum > maximum or maximum > option_width
            for minimum, maximum in zip(
                batch.min_counts,
                batch.max_counts,
                strict=True,
            )
        ):
            raise ValueError("native policy batch select cardinality is invalid")
        tensor_batches = (
            int(batch.states.card_ids.shape[0]),
            int(batch.options.valid_options.shape[0]),
            int(batch.unique_deck_card_ids.shape[0]),
            int(batch.deck_counts.shape[0]),
            int(batch.deck_valid_mask.shape[0]),
        )
        if any(rows != batch_size for rows in tensor_batches):
            raise ValueError("native policy batch model tensors are misaligned")
        deck_shape = batch.unique_deck_card_ids.shape
        if (
            len(deck_shape) != 2
            or batch.deck_counts.shape != deck_shape
            or batch.deck_valid_mask.shape != deck_shape
        ):
            raise ValueError("native exact-deck tensors are misaligned")
        if (
            batch.unique_deck_card_ids.dtype != torch.long
            or not batch.deck_counts.dtype.is_floating_point
            or batch.deck_valid_mask.dtype != torch.bool
        ):
            raise TypeError("native exact-deck tensors use invalid dtypes")
        belief_rows = (
            int(batch.belief_summary.row_indices.numel())
            if batch.belief_summary.row_indices is not None
            else int(batch.belief_summary.card_ids.shape[0])
        )
        if belief_rows != batch_size:
            raise ValueError("native policy batch belief rows are misaligned")
        tensors: list[Tensor] = [
            batch.unique_deck_card_ids,
            batch.deck_counts,
            batch.deck_valid_mask,
            batch.belief_summary.card_ids,
            batch.belief_summary.expected_counts,
            batch.belief_summary.valid_mask,
            batch.belief_summary.scalars,
        ]
        if batch.belief_summary.row_indices is not None:
            tensors.append(batch.belief_summary.row_indices)
        tensors.extend(_dataclass_tensors(batch.states))
        tensors.extend(_dataclass_tensors(batch.options))
        if any(
            tensor.device.type != self.device.type
            or (
                self.device.index is not None
                and tensor.device.index != self.device.index
            )
            for tensor in tensors
        ):
            raise ValueError("native policy batch tensors differ from executor device")

    def _verify_contract(self) -> None:
        config = self.model.config
        if model_config_fingerprint(config) != self.identity.model_config_fingerprint:
            raise ValueError("native inference model config differs from identity")
        if config.resolved_registry_sha256 != self.identity.exact_registry_fingerprint:
            raise ValueError("native inference exact registry differs from identity")
        if (
            config.public_deck_catalog_fingerprint
            != self.identity.public_deck_catalog_fingerprint
        ):
            raise ValueError("native inference catalog differs from identity")

    def _validate_generator(self, generator: torch.Generator | None) -> None:
        if generator is None:
            return
        generator_device = torch.device(generator.device)
        if generator_device.type != self.device.type:
            raise ValueError("sampling generator must use the executor device type")


def _collate_exact_routes(
    config: SimpleStatelessModelConfig,
) -> tuple[tuple[str, ...], Tensor, Tensor, Tensor]:
    """Build one canonical unique-card row per configured exact route."""
    if not config.exact_routes:
        raise ValueError("native inference requires at least one exact deck route")
    encoded = tuple(
        tuple(
            (int(card_id), sum(1 for _unused in repeated))
            for card_id, repeated in groupby(route.canonical_card_ids)
        )
        for route in config.exact_routes
    )
    width = max(len(row) for row in encoded)
    card_ids = torch.zeros((len(encoded), width), dtype=torch.long)
    counts = torch.zeros((len(encoded), width), dtype=torch.float32)
    valid_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for row_index, row in enumerate(encoded):
        row_width = len(row)
        card_ids[row_index, :row_width] = torch.tensor(
            tuple(card_id for card_id, _count in row),
            dtype=torch.long,
        )
        counts[row_index, :row_width] = torch.tensor(
            tuple(count for _card_id, count in row),
            dtype=torch.float32,
        )
        valid_mask[row_index, :row_width] = True
    return (
        tuple(route.signature for route in config.exact_routes),
        card_ids,
        counts,
        valid_mask,
    )


def _dataclass_tensors(value: StateBatch | OptionBatch) -> Sequence[Tensor]:
    """Return every direct tensor field without reading tensor values."""
    return tuple(
        item
        for field in fields(value)
        if isinstance((item := getattr(value, field.name)), Tensor)
    )


__all__ = [
    "NativePolicyInferenceExecutor",
    "NativePolicyNumpyActionBatch",
    "NativePolicyNumpyTrace",
    "NativePolicyTensorActionBatch",
    "NativePolicyTensorTrace",
    "PolicyInputBatch",
    "TensorNativePolicyInputBatch",
]
