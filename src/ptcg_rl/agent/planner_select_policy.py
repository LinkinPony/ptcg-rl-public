"""PolicyRuntimeAgent adapter for the shared schema-9 v5 planner service."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from typing import Any, Literal, cast

import torch

from ptcg_rl.actions.encoding import StateTokenLayout, encode_option_arrays
from ptcg_rl.agent.search.policy_inputs import build_canonical_policy_input
from ptcg_rl.context import GameContextFeatures, GameContextSnapshot
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.model import collate_encoded_options, collate_state_tokens
from ptcg_rl.model.root_input_fingerprint import (
    canonical_planner_root_input_fingerprint,
)
from ptcg_rl.model.state_encoder import encode_observation_token_arrays
from ptcg_rl.rl.collection import ModelRolloutPolicy
from ptcg_rl.rl.planner_behavior_policy_contract import PlannerPolicyDecision
from ptcg_rl.rl.planner_behavior_service import PlannerBehaviorService
from ptcg_rl.rl.planner_inference_session import (
    PlannerInferenceLease,
    PlannerInferenceSession,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeConfig
from ptcg_rl.rl.planner_service_inputs import PlannerRootRow


class PlannerSelectPolicy:
    """Select complete actions through the exact packaged planner hot path."""

    def __init__(
        self,
        *,
        model: Any,
        device: torch.device | str,
        runtime_config: ResolvedPlannerRuntimeConfig,
        service: PlannerBehaviorService,
        policy_version: int = 0,
        proposal_version: int = 1,
        verified_model_fingerprint: str | None = None,
        base_policy: Any | None = None,
        owned_runtime: Any | None = None,
        temperature: float = 0.0,
    ) -> None:
        if temperature < 0.0:
            raise ValueError("planner base temperature must be non-negative")
        self._device = torch.device(device)
        self._runtime = runtime_config
        self._service = service
        self._temperature = float(temperature)
        self._base_policy = base_policy
        self._owned_runtime = owned_runtime
        self._closed = False
        self._policy = ModelRolloutPolicy(
            model,
            policy_version=policy_version,
            planner_context_capacity=runtime_config.contexts.retained_root_rows,
            verified_model_fingerprint=verified_model_fingerprint,
            proposal_version=proposal_version,
        )
        self._own_deck: CanonicalDeck | None = None
        self._context_snapshot: GameContextSnapshot | None = None
        self._context_features: GameContextFeatures | None = None
        self._deadline_monotonic: float | None = None
        self.last_decision: PlannerPolicyDecision | None = None

    @property
    def planner_enabled(self) -> bool:
        """Report that this packaged select surface executes schema-9 planning."""
        return True

    @property
    def model_fingerprint(self) -> str:
        """Return the immutable migrated serving-model identity."""
        return self._policy.model_fingerprint

    @property
    def policy_version(self) -> int:
        """Return the checkpoint publication version held by this process."""
        return self._policy.policy_version

    @property
    def proposal_version(self) -> int:
        """Return the learned proposal architecture version."""
        return self._policy.proposal_version

    @property
    def own_deck_signature(self) -> str | None:
        """Expose the bound deck identity for submission validation."""
        return None if self._own_deck is None else self._own_deck.signature

    @property
    def deck_conditioning_enabled(self) -> bool:
        """Delegate checkpoint conditioning status for package diagnostics."""
        return bool(getattr(self._base_policy, "deck_conditioning_enabled", False))

    @property
    def checkpoint_registry_sha256(self) -> str | None:
        """Delegate the checkpoint's immutable deck registry identity."""
        return cast(
            str | None,
            getattr(self._base_policy, "checkpoint_registry_sha256", None),
        )

    @property
    def selected_private_profile_module_key(self) -> str | None:
        """Delegate the selected private route for package diagnostics."""
        return cast(
            str | None,
            getattr(self._base_policy, "selected_private_profile_module_key", None),
        )

    @property
    def packaged_private_profile_count(self) -> int:
        """Delegate packaged private-profile coverage diagnostics."""
        return int(getattr(self._base_policy, "packaged_private_profile_count", 0))

    def prewarm(self) -> None:
        """Prewarm the already loaded base model without planner side effects."""
        prewarm = getattr(self._base_policy, "prewarm", None)
        if callable(prewarm):
            prewarm()

    def close(self) -> None:
        """Drain and close the owned native planner runtime exactly once."""
        if self._closed:
            return
        self._closed = True
        runtime = self._owned_runtime
        self._owned_runtime = None
        if runtime is not None:
            runtime.close()

    def bind_own_deck(self, card_ids: Sequence[int]) -> None:
        """Bind the exact deployment deck for model conditioning and v5 roots."""
        self._own_deck = canonicalize_deck(card_ids)
        bind_base = getattr(self._base_policy, "bind_own_deck", None)
        if callable(bind_base):
            bind_base(self._own_deck.card_ids)

    def bind_runtime_context(
        self,
        *,
        context_snapshot: GameContextSnapshot,
        context_features: GameContextFeatures,
        deadline_monotonic: float,
    ) -> None:
        """Bind one callback's exact online context and absolute deadline."""
        deadline = float(deadline_monotonic)
        if deadline <= time.monotonic():
            raise TimeoutError("packaged planner deadline already expired")
        self.last_decision = None
        self._context_snapshot = context_snapshot
        self._context_features = context_features
        self._deadline_monotonic = deadline

    def configure_inference_cache(self, *, enabled: bool) -> None:
        """Accept the runtime cache hook; planner contexts have explicit scope."""
        del enabled

    def clear_inference_cache(self) -> None:
        """Clear callback bindings after every PolicyRuntimeAgent invocation."""
        self._context_snapshot = None
        self._context_features = None
        self._deadline_monotonic = None

    def reset_decision_telemetry(self) -> None:
        """Clear any decision retained from an earlier callback."""
        self.last_decision = None

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Decode a base root, run shared v5 planning, and release the context."""
        if self._closed:
            raise RuntimeError("packaged planner policy is closed")
        if self._own_deck is None:
            raise RuntimeError("packaged planner has no bound own deck")
        if (
            self._context_snapshot is None
            or self._context_features is None
            or self._deadline_monotonic is None
        ):
            raise RuntimeError("packaged planner has no bound callback context")
        prepared = build_canonical_policy_input(observation)
        if prepared is None:
            raise ValueError("packaged planner observation cannot be tensorized")
        layout = StateTokenLayout.from_observation(
            observation,
            context_features=self._context_features,
        )
        state_features = encode_observation_token_arrays(
            observation,
            layout=layout,
        )
        option_features = encode_option_arrays(
            observation.get("select")
            if isinstance(observation, Mapping)
            else getattr(observation, "select", None),
            layout,
        )
        min_count = min(len(option_features), max(0, prepared.min_count))
        max_count = min(
            len(option_features),
            max(min_count, prepared.max_count),
        )
        states = replace(
            collate_state_tokens((state_features,), device=self._device),
            root_input_fingerprints=(
                canonical_planner_root_input_fingerprint(
                    state_features,
                    option_features,
                    min_count=min_count,
                    max_count=max_count,
                ),
            ),
        )
        options = collate_encoded_options(
            (option_features,),
            min_counts=(min_count,),
            max_counts=(max_count,),
            device=self._device,
        )
        decks = DeckBatch.from_decks((self._own_deck,), device=self._device)
        trace = self._policy.sample_decode_with_trace_for_request(
            states,
            options,
            decks,
            temperature=self._temperature,
            model_version_lease=None,
            retain_planner_context=True,
        )
        if trace.planner_fallback_reason:
            if trace.planner_context_handles:
                raise RuntimeError("packaged base fallback retained a context")
            return trace.actions[0]
        if len(trace.planner_context_handles) != 1:
            raise RuntimeError("packaged root decode did not retain one context")
        session: PlannerInferenceSession | None = None
        try:
            tensor_schema_fingerprint = (
                self._runtime.resolve_static().tensor_schema_fingerprint
            )
            self._policy.bind_planner_context_handles(
                trace.planner_context_handles,
                policy_version=self._policy.policy_version,
                tensor_schema_fingerprint=tensor_schema_fingerprint,
            )
            session = PlannerInferenceSession(
                policy=self._policy,
                states=states,
                options=options,
                decks=decks,
                context_handles=trace.planner_context_handles,
                lease=PlannerInferenceLease(
                    model_fingerprint=self._policy.model_fingerprint,
                    policy_version=self._policy.policy_version,
                    tensor_schema_fingerprint=tensor_schema_fingerprint,
                    deadline_monotonic=self._deadline_monotonic,
                    inference_device_type=cast(
                        Literal["cpu", "cuda"],
                        self._policy.planner_inference_device_type,
                    ),
                    inference_timeout_seconds=(
                        self._runtime.deadlines.inference_timeout_seconds
                    ),
                ),
            )
            row = PlannerRootRow(
                row_id="packaged-act",
                seat=_root_player(observation),
                observation=observation,
                context_features=self._context_features,
                context_snapshot=self._context_snapshot,
                own_deck=self._own_deck.card_ids,
                base_action=trace.actions[0],
                base_old_logprob=float(trace.action_logprobs[0].item()),
            )
            decision = self._service.plan_runtime_row(
                policy=self._policy,
                row=row,
                inference_session=session,
            )
        finally:
            if session is None:
                with suppress(Exception):
                    self._policy.release_planner_context_handles(
                        trace.planner_context_handles
                    )
            else:
                with suppress(Exception):
                    session.release_rows(
                        (0,),
                        deadline_monotonic=(
                            time.monotonic()
                            + self._runtime.deadlines.cleanup_timeout_seconds
                        ),
                    )
        self.last_decision = decision
        return decision.action


def _root_player(observation: Any) -> int:
    current = (
        observation.get("current")
        if isinstance(observation, Mapping)
        else getattr(observation, "current", None)
    )
    player = (
        current.get("yourIndex", 0)
        if isinstance(current, Mapping)
        else getattr(current, "yourIndex", 0)
    )
    resolved = int(player)
    if resolved not in (0, 1):
        raise ValueError("packaged planner root player must be 0 or 1")
    return resolved


__all__ = ["PlannerSelectPolicy"]
