"""Protected root materialization and exact one-action native reanalysis."""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import orjson

from ptcg_rl.actions.encoding import EncodedOptionInput
from ptcg_rl.agent.search.policy_inputs import build_canonical_policy_input
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    GameContext,
    GameContextSnapshot,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.context.belief import opponent_belief_state_from_evidence
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.consequence_identity import (
    candidate_action_fingerprint,
    hidden_information_fingerprint,
)
from ptcg_rl.engine.native_consequence import NativeConsequenceLane
from ptcg_rl.engine.native_consequence_payload import (
    NativeConsequenceEndpoint,
    NativeConsequenceMetadataColumn,
)
from ptcg_rl.engine.session import HiddenInformation
from ptcg_rl.model.state_encoder import StateTokenInput
from ptcg_rl.rl.amortized_policy_iteration.candidate_proposal import (
    CandidateProposal,
)
from ptcg_rl.rl.amortized_policy_iteration.contracts import (
    AmortizedPolicyIterationConfig,
    BehaviorKind,
)

_ROOT_ID_DOMAIN = b"ptcg-rl/api-root/v1\x00"
_PRODUCER_DOMAIN = b"ptcg-rl/api-native-producer/v1\x00"


@dataclass(frozen=True, slots=True)
class StudentReanalysisRoot:
    """Actor-visible root fields that are safe for model collation."""

    root_id: str
    game_id: str
    seat: int
    decision_index: int
    state: StateTokenInput
    options: EncodedOptionInput
    min_count: int
    max_count: int
    deck: CanonicalDeck
    behavior_kind: BehaviorKind
    behavior_action: tuple[int, ...]
    behavior_probability: float
    sampling_temperature: float
    policy_version: int

    def __post_init__(self) -> None:
        if not self.root_id or not self.game_id:
            raise ValueError("reanalysis root identities must be non-empty")
        if self.seat not in (0, 1) or self.decision_index < 0:
            raise ValueError("reanalysis root decision identity is invalid")
        if not 0 <= self.min_count <= self.max_count <= len(self.options):
            raise ValueError("reanalysis root action bounds are invalid")
        if not math.isfinite(self.behavior_probability) or not (
            0.0 < self.behavior_probability <= 1.0
        ):
            raise ValueError("reanalysis behavior probability must be in (0, 1]")
        if (
            not math.isfinite(self.sampling_temperature)
            or self.sampling_temperature <= 0
        ):
            raise ValueError("reanalysis sampling temperature must be positive")
        if self.policy_version < 0:
            raise ValueError("reanalysis root policy version must be non-negative")


@dataclass(frozen=True, slots=True)
class ProtectedReanalysisRoot:
    """Engine-only material that must never enter root model collation."""

    observation_json: bytes
    context_snapshots: tuple[GameContextSnapshot, GameContextSnapshot]
    deck_pair: tuple[CanonicalDeck, CanonicalDeck]

    def __post_init__(self) -> None:
        if not self.observation_json:
            raise ValueError("protected root observation must not be empty")
        if tuple(snapshot.player_index for snapshot in self.context_snapshots) != (
            0,
            1,
        ):
            raise ValueError("protected root contexts must be ordered by seat")

    def observation(self) -> Mapping[str, Any]:
        """Decode the immutable actor observation for CPU-only reanalysis."""
        raw = orjson.loads(self.observation_json)
        if not isinstance(raw, dict):
            raise ValueError("protected root observation is not an object")
        return raw


@dataclass(frozen=True, slots=True)
class ReanalysisRoot:
    """Separately typed student and protected views of one archived root."""

    student: StudentReanalysisRoot
    protected: ProtectedReanalysisRoot

    def __post_init__(self) -> None:
        if self.student.deck != self.protected.deck_pair[self.student.seat]:
            raise ValueError("student root deck differs from protected acting deck")


@dataclass(frozen=True, slots=True)
class NativeReanalysisJob:
    """One learner-proposed root sent to a persistent CPU engine worker."""

    root: ReanalysisRoot
    proposal: CandidateProposal
    proposal_policy_version: int
    producer_contract_fingerprint: bytes
    stochastic_seed: int

    def __post_init__(self) -> None:
        if self.proposal_policy_version < 0:
            raise ValueError("proposal policy version must be non-negative")
        if len(self.producer_contract_fingerprint) != 32:
            raise ValueError("native producer contract must contain 32 bytes")


@dataclass(frozen=True, slots=True)
class NativeReanalysisJobBatch:
    """One queue envelope containing independently reproducible native jobs.

    The native engine ABI remains root-oriented.  Batching only amortizes the
    multiprocessing queue framing and pickle calls; workers execute the jobs in
    order through the exact single-root implementation below.
    """

    jobs: tuple[NativeReanalysisJob, ...]

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("native reanalysis job batch must not be empty")


@dataclass(frozen=True, slots=True)
class LeafActorInput:
    """One actor-visible successor input before learner-side H200 collation."""

    state: StateTokenInput
    options: EncodedOptionInput
    min_count: int
    max_count: int
    deck: CanonicalDeck
    observation_json: bytes


@dataclass(frozen=True, slots=True)
class NativeReanalysisCell:
    """One protected candidate-by-world native consequence."""

    candidate_index: int
    world_index: int
    candidate_id: str
    particle_id: str
    sampling_weight: float
    endpoint: NativeConsequenceEndpoint
    error: int
    root_player: int
    leaf_player: int | None
    engine_result: int | None
    transition_steps: int
    forced_steps: int
    leaf: LeafActorInput | None


@dataclass(frozen=True, slots=True)
class NativeReanalysisResult:
    """Compact worker result with no materialized hidden zones."""

    root: ReanalysisRoot
    proposal: CandidateProposal
    proposal_policy_version: int
    cells: tuple[NativeReanalysisCell, ...]
    engine_library_fingerprint: str
    native_abi_fingerprint: str
    native_seconds: float
    error_message: str = ""


def freeze_reanalysis_root(
    *,
    game_id: str,
    seat: int,
    decision_index: int,
    state: StateTokenInput,
    options: EncodedOptionInput,
    min_count: int,
    max_count: int,
    deck_pair: Sequence[Sequence[int]],
    observation: Mapping[str, Any],
    context_snapshots: tuple[GameContextSnapshot, GameContextSnapshot],
    behavior_kind: BehaviorKind,
    behavior_action: Sequence[int],
    behavior_logprob: float,
    sampling_temperature: float,
    policy_version: int,
) -> ReanalysisRoot:
    """Freeze one pre-action root with an auditable information-set split."""
    canonical_decks = tuple(canonicalize_deck(deck) for deck in deck_pair)
    if len(canonical_decks) != 2:
        raise ValueError("reanalysis root requires exactly two decks")
    root_id = reanalysis_root_id(
        game_id=game_id,
        seat=seat,
        decision_index=decision_index,
    )
    observation_json = orjson.dumps(
        observation,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
    )
    student = StudentReanalysisRoot(
        root_id=root_id,
        game_id=game_id,
        seat=seat,
        decision_index=decision_index,
        state=state.without_layout(),
        options=options,
        min_count=min_count,
        max_count=max_count,
        deck=canonical_decks[seat],
        behavior_kind=behavior_kind,
        behavior_action=tuple(int(value) for value in behavior_action),
        behavior_probability=math.exp(float(behavior_logprob)),
        sampling_temperature=float(sampling_temperature),
        policy_version=policy_version,
    )
    return ReanalysisRoot(
        student=student,
        protected=ProtectedReanalysisRoot(
            observation_json=observation_json,
            context_snapshots=context_snapshots,
            deck_pair=(canonical_decks[0], canonical_decks[1]),
        ),
    )


def rebind_reanalysis_game_id(root: ReanalysisRoot, *, game_id: str) -> ReanalysisRoot:
    """Namespace an actor-local root identity before shared-queue publication."""
    student = replace(
        root.student,
        root_id=reanalysis_root_id(
            game_id=game_id,
            seat=root.student.seat,
            decision_index=root.student.decision_index,
        ),
        game_id=game_id,
    )
    return replace(root, student=student)


def reanalysis_root_id(*, game_id: str, seat: int, decision_index: int) -> str:
    """Return a stable root identity without embedding protected state."""
    digest = hashlib.sha256()
    digest.update(_ROOT_ID_DOMAIN)
    digest.update(game_id.encode("utf-8"))
    digest.update(f"\x00{seat}\x00{decision_index}".encode("ascii"))
    return digest.hexdigest()


def native_producer_fingerprint(
    config: AmortizedPolicyIterationConfig,
) -> bytes:
    """Bind engine evidence to the complete resolved reanalysis contract."""
    payload = orjson.dumps(
        config.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(_PRODUCER_DOMAIN + payload).digest()


def execute_native_reanalysis_job(
    job: NativeReanalysisJob,
    *,
    config: AmortizedPolicyIterationConfig,
    belief_sampler: BeliefSampler,
    belief_feature_producer: OpponentBeliefFeatureProducer | None,
    lane: NativeConsequenceLane,
) -> NativeReanalysisResult:
    """Sample shared worlds, execute one exact action, and tensorize leaf views."""
    started = time.perf_counter()
    try:
        determinizations = _sample_determinizations(
            job.root,
            sampler=belief_sampler,
            count=config.belief_worlds,
            seed=job.stochastic_seed,
        )
        candidate_actions = tuple(
            candidate.action for candidate in job.proposal.candidates
        )
        observation = job.root.protected.observation()
        state_token = observation.get("search_begin_input")
        if not isinstance(state_token, (str, bytes)) or not state_token:
            raise ValueError("reanalysis root has no engine state token")
        native = lane.run(
            state_token,
            hidden_worlds=tuple(item.hidden for item in determinizations),
            candidate_actions=candidate_actions,
            producer_contract_fingerprint=job.producer_contract_fingerprint,
            root_player=job.root.student.seat,
            manual_coin=False,
            stochastic_seed=job.stochastic_seed,
            max_cells=config.native.max_cells_per_root,
            max_engine_steps=config.native.max_engine_steps_per_root,
            max_forced_steps=config.native.max_forced_steps,
            max_observation_bytes=config.native.max_observation_bytes,
        )
        cells = _materialize_cells(
            job,
            native=native,
            determinizations=determinizations,
            belief_feature_producer=belief_feature_producer,
        )
        return NativeReanalysisResult(
            root=job.root,
            proposal=job.proposal,
            proposal_policy_version=job.proposal_policy_version,
            cells=cells,
            engine_library_fingerprint=lane.engine_library_fingerprint,
            native_abi_fingerprint=lane.native_abi_fingerprint,
            native_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return NativeReanalysisResult(
            root=job.root,
            proposal=job.proposal,
            proposal_policy_version=job.proposal_policy_version,
            cells=(),
            engine_library_fingerprint=lane.engine_library_fingerprint,
            native_abi_fingerprint=lane.native_abi_fingerprint,
            native_seconds=time.perf_counter() - started,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def _sample_determinizations(
    root: ReanalysisRoot,
    *,
    sampler: BeliefSampler,
    count: int,
    seed: int,
) -> tuple[Any, ...]:
    observation = root.protected.observation()
    evidence = extract_observation_evidence(observation)
    context = root.protected.context_snapshots[root.student.seat]
    context_features = GameContext.from_snapshot(context).features(observation)
    opponent_state = opponent_belief_state_from_evidence(
        evidence,
        context_features,
        context_snapshot=context,
    )
    rng = random.Random(seed)
    return tuple(
        sampler.sample_from_evidence(
            evidence,
            your_deck=root.student.deck.card_ids,
            opponent_state=opponent_state,
            rng=rng,
        )
        for _ in range(count)
    )


def _materialize_cells(
    job: NativeReanalysisJob,
    *,
    native: Any,
    determinizations: Sequence[Any],
    belief_feature_producer: OpponentBeliefFeatureProducer | None,
) -> tuple[NativeReanalysisCell, ...]:
    cells = []
    worlds = len(determinizations)
    weight = 1.0 / float(worlds)
    for candidate_index, candidate in enumerate(job.proposal.candidates):
        candidate_id = candidate_action_fingerprint(candidate.action)
        for world_index, determinization in enumerate(determinizations):
            row = native.payload.row_index(world_index, candidate_index)
            metadata = native.metadata[row]
            error = int(metadata[int(NativeConsequenceMetadataColumn.ERROR)])
            endpoint = NativeConsequenceEndpoint(
                int(metadata[int(NativeConsequenceMetadataColumn.ENDPOINT)])
            )
            leaf_player_raw = int(
                metadata[int(NativeConsequenceMetadataColumn.LEAF_PLAYER)]
            )
            leaf_player = leaf_player_raw if leaf_player_raw in (0, 1) else None
            leaf = None
            if error == 0 and endpoint is not NativeConsequenceEndpoint.TERMINAL:
                leaf_observation = native.decode_leaf_observation_row(row)
                if leaf_observation is None or leaf_player is None:
                    error = 1
                    endpoint = NativeConsequenceEndpoint.INVALID
                else:
                    leaf = _leaf_actor_input(
                        job.root,
                        leaf_observation=leaf_observation,
                        leaf_player=leaf_player,
                        determinization=determinization,
                        belief_feature_producer=belief_feature_producer,
                    )
            engine_result_raw = int(
                metadata[int(NativeConsequenceMetadataColumn.RESULT)]
            )
            cells.append(
                NativeReanalysisCell(
                    candidate_index=candidate_index,
                    world_index=world_index,
                    candidate_id=candidate_id,
                    particle_id=_particle_id(
                        determinization.hidden,
                        world_index=world_index,
                    ),
                    sampling_weight=weight,
                    endpoint=endpoint,
                    error=error,
                    root_player=job.root.student.seat,
                    leaf_player=leaf_player,
                    engine_result=(
                        engine_result_raw if engine_result_raw in (0, 1, 2) else None
                    ),
                    transition_steps=int(
                        metadata[int(NativeConsequenceMetadataColumn.TRANSITION_STEPS)]
                    ),
                    forced_steps=int(
                        metadata[int(NativeConsequenceMetadataColumn.FORCED_STEPS)]
                    ),
                    leaf=leaf,
                )
            )
    return tuple(cells)


def _leaf_actor_input(
    root: ReanalysisRoot,
    *,
    leaf_observation: Mapping[str, Any],
    leaf_player: int,
    determinization: Any,
    belief_feature_producer: OpponentBeliefFeatureProducer | None,
) -> LeafActorInput:
    deck = _leaf_deck(root, leaf_player=leaf_player, determinization=determinization)
    context = GameContext.from_snapshot(root.protected.context_snapshots[leaf_player])
    context.set_own_deck(deck.card_ids)
    observation = dict(leaf_observation)
    features = context.update(observation)
    if belief_feature_producer is not None:
        belief, entropy, empty = belief_feature_producer.features(
            observation,
            features,
        )
        features = replace(
            features,
            opponent_belief=belief,
            opponent_belief_entropy=entropy,
            opponent_belief_empty=empty,
        )
    observation["gameContext"] = features.as_observation_dict()
    policy_input = build_canonical_policy_input(observation)
    if policy_input is None:
        raise ValueError("native successor has no canonical policy input")
    return LeafActorInput(
        state=policy_input.state.without_layout(),
        options=policy_input.options,
        min_count=policy_input.min_count,
        max_count=policy_input.max_count,
        deck=deck,
        observation_json=orjson.dumps(
            observation,
            option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        ),
    )


def _leaf_deck(
    root: ReanalysisRoot,
    *,
    leaf_player: int,
    determinization: Any,
) -> CanonicalDeck:
    if leaf_player == root.student.seat:
        return root.student.deck
    counts: Counter[int] = Counter(determinization.opponent_deck_counts)
    cards = tuple(
        card_id for card_id, count in sorted(counts.items()) for _ in range(count)
    )
    return canonicalize_deck(cards)


def _particle_id(hidden: HiddenInformation, *, world_index: int) -> str:
    """Disambiguate IID duplicate worlds without exposing either identity."""
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(hidden_information_fingerprint(hidden)))
    digest.update(world_index.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def native_reanalysis_worker(
    *,
    config_data: Mapping[str, Any],
    rollout_belief_data: Mapping[str, Any],
    job_queue: Any,
    result_queue: Any,
    library_path: str | None,
) -> None:
    """Run a persistent CPU-only native lane until the parent terminates it."""
    from ptcg_rl.rl.rollout import RolloutBeliefConfig

    config = AmortizedPolicyIterationConfig.model_validate(config_data)
    rollout_belief = RolloutBeliefConfig.model_validate(rollout_belief_data)
    sampler = BeliefSampler(config=config.belief_sampler)
    belief_features = (
        OpponentBeliefFeatureProducer.from_config(rollout_belief)
        if rollout_belief.enabled
        else None
    )
    with NativeConsequenceLane(
        library_path=None if library_path is None else Path(library_path)
    ) as lane:
        while True:
            wire_item = job_queue.get()
            if wire_item is None:
                return
            for job in native_reanalysis_jobs_from_queue_item(wire_item):
                result_queue.put(
                    execute_native_reanalysis_job(
                        job,
                        config=config,
                        belief_sampler=sampler,
                        belief_feature_producer=belief_features,
                        lane=lane,
                    )
                )


def native_reanalysis_jobs_from_queue_item(
    wire_item: object,
) -> tuple[NativeReanalysisJob, ...]:
    """Normalize both legacy single-job and batched queue wire contracts."""
    if isinstance(wire_item, NativeReanalysisJob):
        return (wire_item,)
    if isinstance(wire_item, NativeReanalysisJobBatch) and all(
        isinstance(job, NativeReanalysisJob) for job in wire_item.jobs
    ):
        return wire_item.jobs
    raise TypeError("native reanalysis queue received an invalid job")


__all__ = [
    "LeafActorInput",
    "NativeReanalysisCell",
    "NativeReanalysisJob",
    "NativeReanalysisJobBatch",
    "NativeReanalysisResult",
    "ProtectedReanalysisRoot",
    "ReanalysisRoot",
    "StudentReanalysisRoot",
    "execute_native_reanalysis_job",
    "freeze_reanalysis_root",
    "native_producer_fingerprint",
    "native_reanalysis_jobs_from_queue_item",
    "native_reanalysis_worker",
    "rebind_reanalysis_game_id",
    "reanalysis_root_id",
]
