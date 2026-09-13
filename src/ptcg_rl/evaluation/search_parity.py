"""Replay-root serving/Search API feature-parity and leakage audit."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.actions.selection import is_forced
from ptcg_rl.agent.probe import (
    ActTimeSearchConfig,
    enumerate_select_actions,
    observation_with_probe_features,
    run_runtime_probe_features,
)
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.context import (
    observation_with_context,
    public_search_observation,
)
from ptcg_rl.agent.search.policy_inputs import canonical_inputs_bitwise_equal
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.context import (
    GameContext,
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
    opponent_belief_state_from_evidence,
)
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.records import (
    DEFAULT_CHUNK_SIZE,
    iter_replay_steps,
    replay_stub,
)
from ptcg_rl.engine.session import HiddenInformation, SearchSession
from ptcg_rl.model.state_encoder import TOKEN_KIND_OOV_INDEX

_REQUIRED_CONTEXT_FIELDS = frozenset(
    {
        "ownUnseen",
        "opponentRevealed",
        "opponentBelief",
        "opponentBeliefEntropy",
        "opponentBeliefEmpty",
        "historyCounts",
        "deckFlowCounts",
        "lastAttacks",
    }
)


class SearchFeatureParityConfig(BaseModel):
    """Hydra-backed P0 root-parity audit configuration."""

    model_config = ConfigDict(extra="forbid")

    replay_paths: tuple[Path, ...] = ()
    replay_glob: str = (
        "outputs/kaggle_submission_replays/54498922_comfey_v12395/*.json"
    )
    team_name: str | None = "Marshall Maximizer"
    seat_index: int | None = None
    deck_path: Path = Path(
        "docs/experiments/rl_dynamic_deck_pool_20260708/decks/"
        "29_comfey_yveltal_shaymin_4f8e151b4dd0.csv"
    )
    checkpoint_path: Path = Path(
        "outputs/inference_time_search/p0/assets/policy_v12395.pt"
    )
    device: str = "cpu"
    belief: OpponentBeliefFeatureConfig = Field(
        default_factory=OpponentBeliefFeatureConfig
    )
    sampler: BeliefSamplerConfig = Field(
        default_factory=lambda: BeliefSamplerConfig(mode="archetype")
    )
    probe: ActTimeSearchConfig = Field(default_factory=ActTimeSearchConfig)
    include_probe_features: bool = True
    max_replays: int | None = 112
    min_roots: int = 500
    max_roots: int = 1_000
    roots_per_phase_per_replay: int = 2
    early_turn_max: int = 3
    mid_turn_max: int = 9
    max_complete_actions: int = 64
    top_k: int = 4
    seed: int = 0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    output_parquet: Path = Path(
        "outputs/inference_time_search/p0/feature_parity/roots.parquet"
    )
    output_summary: Path = Path(
        "outputs/inference_time_search/p0/feature_parity/summary.json"
    )
    compression: str = "zstd"
    logit_atol: float = 1.0e-5
    value_atol: float = 1.0e-5
    prior_kl_atol: float = 1.0e-7

    @field_validator(
        "min_roots",
        "max_roots",
        "roots_per_phase_per_replay",
        "max_complete_actions",
        "top_k",
        "chunk_size",
    )
    @classmethod
    def positive_limits(cls, value: int) -> int:
        """Reject non-positive audit limits."""
        if value <= 0:
            raise ValueError("parity audit limits must be positive")
        return value

    @field_validator("max_replays")
    @classmethod
    def optional_positive(cls, value: int | None) -> int | None:
        """Reject non-positive replay limits."""
        if value is not None and value <= 0:
            raise ValueError("max_replays must be positive")
        return value

    @field_validator("seat_index")
    @classmethod
    def valid_seat(cls, value: int | None) -> int | None:
        """Restrict explicit seat selection."""
        if value is not None and value not in (0, 1):
            raise ValueError("seat_index must be 0 or 1")
        return value

    @model_validator(mode="after")
    def valid_root_and_phase_bounds(self) -> SearchFeatureParityConfig:
        """Require coherent root and turn-phase limits."""
        if self.max_roots < self.min_roots:
            raise ValueError("max_roots must be >= min_roots")
        if self.mid_turn_max < self.early_turn_max:
            raise ValueError("mid_turn_max must be >= early_turn_max")
        return self


class _ParityStats:
    def __init__(self) -> None:
        self.roots = 0
        self.replays = 0
        self.bitwise_matches = 0
        self.top1_matches = 0
        self.hidden_matches = 0
        self.unknown_drift_roots = 0
        self.context_missing_roots = 0
        self.max_logit_abs_error = 0.0
        self.max_value_abs_error = 0.0
        self.max_prior_kl = 0.0
        self.min_top_k_recall = 1.0
        self.phase_counts: Counter[str] = Counter()
        self.source_hash = hashlib.sha256()


def run_search_feature_parity_audit(
    config: SearchFeatureParityConfig,
) -> dict[str, Any]:
    """Stream replay roots and verify serving/Search API network equivalence."""
    replay_paths = _resolve_replay_paths(config)
    deck = records.read_deck(records.repo_path(config.deck_path))
    checkpoint_path = records.repo_path(config.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    policy = CheckpointPolicy(
        checkpoint_path,
        device=config.device,
        own_deck=deck,
    )
    belief = OpponentBeliefFeatureProducer.from_config(
        _resolve_belief_config(config.belief)
    )
    sampler = BeliefSampler(config=_resolve_sampler_config(config.sampler))
    rng = random.Random(config.seed)
    stats = _ParityStats()
    output_path = records.repo_path(config.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = pq.ParquetWriter(
        temporary_path,
        _parity_schema(),
        compression=config.compression,
    )
    writer_open = True
    rows: list[dict[str, Any]] = []
    try:
        for replay_path in replay_paths:
            if stats.roots >= config.max_roots:
                break
            stats.replays += 1
            stats.source_hash.update(replay_path.name.encode("utf-8"))
            stats.source_hash.update(_file_sha256(replay_path).encode("ascii"))
            replay_rows = _audit_replay(
                replay_path,
                config=config,
                deck=deck,
                policy=policy,
                belief=belief,
                sampler=sampler,
                rng=rng,
                stats=stats,
            )
            rows.extend(replay_rows)
            if len(rows) >= 128:
                writer.write_table(pa.Table.from_pylist(rows, schema=_parity_schema()))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=_parity_schema()))
            rows.clear()
        writer.close()
        writer_open = False
        temporary_path.replace(output_path)
    finally:
        if writer_open:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()

    summary = _parity_summary(config, policy, stats, output_path)
    _write_json_atomic(records.repo_path(config.output_summary), summary)
    return summary


def _audit_replay(
    replay_path: Path,
    *,
    config: SearchFeatureParityConfig,
    deck: Sequence[int],
    policy: CheckpointPolicy,
    belief: OpponentBeliefFeatureProducer,
    sampler: BeliefSampler,
    rng: random.Random,
    stats: _ParityStats,
) -> list[dict[str, Any]]:
    metadata = replay_stub(replay_path, chunk_size=config.chunk_size)
    seat = _resolve_seat(metadata, config)
    episode_id = int(_mapping(metadata.get("info")).get("EpisodeId", replay_path.stem))
    context = GameContext(player_index=seat)
    context.set_own_deck(deck)
    selected_per_phase: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for step_index, sides in iter_replay_steps(
        replay_path,
        chunk_size=config.chunk_size,
    ):
        if stats.roots >= config.max_roots or seat >= len(sides):
            break
        side = sides[seat]
        observation = _mapping(side.get("observation"))
        if str(side.get("status", "")) != "ACTIVE" or not _is_own_decision(
            observation,
            seat,
        ):
            continue
        context_features = context.update(observation)
        select = _mapping(observation.get("select"))
        if not select or is_forced(select):
            continue
        phase = _phase(observation, config)
        if selected_per_phase[phase] >= config.roots_per_phase_per_replay:
            continue
        selected_per_phase[phase] += 1
        enriched_features = belief.augment(observation, context_features)
        serving_observation = observation_with_context(observation, enriched_features)
        row = _audit_root(
            serving_observation,
            episode_id=episode_id,
            step_index=step_index,
            seat=seat,
            phase=phase,
            deck=deck,
            context_features=enriched_features,
            policy=policy,
            sampler=sampler,
            config=config,
            rng=rng,
        )
        rows.append(row)
        _update_stats(stats, row)
    return rows


def _audit_root(
    serving_observation: Mapping[str, Any],
    *,
    episode_id: int,
    step_index: int,
    seat: int,
    phase: str,
    deck: Sequence[int],
    context_features: Any,
    policy: CheckpointPolicy,
    sampler: BeliefSampler,
    config: SearchFeatureParityConfig,
    rng: random.Random,
) -> dict[str, Any]:
    opponent_card_probs, opponent_hand_weights = _belief_distributions(
        policy,
        serving_observation,
        sampler,
    )
    probe_result = None
    if config.include_probe_features:
        probe_result = run_runtime_probe_features(
            serving_observation,
            context_features,
            your_deck=deck,
            sampler=sampler,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
            rng=rng,
            config=config.probe,
        )
        if probe_result is not None:
            serving_observation = observation_with_probe_features(
                serving_observation,
                probe_result,
            )

    evidence = extract_observation_evidence(serving_observation)
    opponent_state = opponent_belief_state_from_evidence(evidence, context_features)
    hidden = sampler.sample_from_evidence(
        evidence,
        your_deck=deck,
        opponent_state=opponent_state,
        opponent_card_probs=opponent_card_probs,
        opponent_hand_weights=opponent_hand_weights,
        rng=rng,
    ).hidden
    adapter_observation = _search_root_observation(
        serving_observation,
        hidden=hidden,
        context_features=context_features,
        probe_result=probe_result,
    )
    permuted_observation = _search_root_observation(
        serving_observation,
        hidden=_permuted_hidden(hidden),
        context_features=context_features,
        probe_result=probe_result,
    )

    serving_input = policy.canonical_input(serving_observation)
    adapter_input = policy.canonical_input(adapter_observation)
    permuted_input = policy.canonical_input(permuted_observation)
    if serving_input is None or adapter_input is None or permuted_input is None:
        raise ValueError("non-forced root produced no canonical policy input")
    bitwise_match = canonical_inputs_bitwise_equal(serving_input, adapter_input)
    hidden_match = canonical_inputs_bitwise_equal(adapter_input, permuted_input)
    serving_logits = policy.first_step_logits(serving_observation)
    adapter_logits = policy.first_step_logits(adapter_observation)
    logit_error = _max_abs_error(serving_logits, adapter_logits)

    actions = enumerate_select_actions(
        serving_observation.get("select"),
        max_actions=config.max_complete_actions,
    )
    serving_priors = policy.action_priors(serving_observation, actions)
    adapter_priors = policy.action_priors(adapter_observation, actions)
    prior_kl = _prior_kl(serving_priors, adapter_priors)
    serving_top = _top_actions(serving_priors, config.top_k)
    adapter_top = _top_actions(adapter_priors, config.top_k)
    top_k_recall = (
        len(set(serving_top) & set(adapter_top)) / float(len(serving_top))
        if serving_top
        else 1.0
    )
    serving_greedy = policy.select_action(serving_observation)
    adapter_greedy = policy.select_action(adapter_observation)
    root_player = _int_field(serving_observation.get("current"), "yourIndex", seat)
    serving_value = policy.value(serving_observation, root_player)
    adapter_value = policy.value(adapter_observation, root_player)
    context_dict = _mapping(adapter_observation.get("gameContext"))
    missing_context = sorted(_REQUIRED_CONTEXT_FIELDS - context_dict.keys())
    token_kinds = Counter(int(kind) for kind in adapter_input.state.token_kinds)
    return {
        "episode_id": episode_id,
        "step_index": step_index,
        "seat": seat,
        "phase": phase,
        "turn": _int_field(serving_observation.get("current"), "turn", -1),
        "select_context": _int_field(serving_observation.get("select"), "context", -1),
        "option_count": len(adapter_input.options),
        "candidate_action_count": len(actions),
        "token_count": len(adapter_input.state.card_ids),
        "token_kind_ids": sorted(token_kinds),
        "token_kind_counts": [token_kinds[key] for key in sorted(token_kinds)],
        "state_option_bitwise_match": bitwise_match,
        "hidden_permutation_match": hidden_match,
        "root_top1_match": serving_greedy == adapter_greedy,
        "root_top_k_recall": top_k_recall,
        "max_logit_abs_error": logit_error,
        "complete_action_prior_kl": prior_kl,
        "serving_value": serving_value,
        "adapter_value": adapter_value,
        "value_abs_error": abs(serving_value - adapter_value),
        "context_missing_fields": missing_context,
        "unknown_token_kind_count": token_kinds.get(TOKEN_KIND_OOV_INDEX, 0),
        "probe_feature_count": sum(probe_result.masks) if probe_result else 0,
    }


def _search_root_observation(
    serving_observation: Mapping[str, Any],
    *,
    hidden: HiddenInformation,
    context_features: Any,
    probe_result: Any,
) -> Mapping[str, Any]:
    with SearchSession.begin(serving_observation, hidden) as session:
        public_root = public_search_observation(
            session.root.observation,
            root_reference=serving_observation,
        )
        adapter = observation_with_context(public_root, context_features)
    if probe_result is not None:
        adapter = observation_with_probe_features(adapter, probe_result)
    return adapter


def _belief_distributions(
    policy: CheckpointPolicy,
    observation: Mapping[str, Any],
    sampler: BeliefSampler,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    if sampler.config.mode != "model":
        return None, None
    distributions = policy.belief_distributions(observation)
    if distributions is None:
        return None, None
    return distributions


def _permuted_hidden(hidden: HiddenInformation) -> HiddenInformation:
    return HiddenInformation(
        your_deck=hidden.your_deck,
        your_prize=hidden.your_prize,
        opponent_deck=tuple(reversed(hidden.opponent_deck)),
        opponent_prize=tuple(reversed(hidden.opponent_prize)),
        opponent_hand=tuple(reversed(hidden.opponent_hand)),
        opponent_active=tuple(reversed(hidden.opponent_active)),
    )


def _update_stats(stats: _ParityStats, row: Mapping[str, Any]) -> None:
    stats.roots += 1
    stats.phase_counts[str(row["phase"])] += 1
    stats.bitwise_matches += int(bool(row["state_option_bitwise_match"]))
    stats.top1_matches += int(bool(row["root_top1_match"]))
    stats.hidden_matches += int(bool(row["hidden_permutation_match"]))
    stats.unknown_drift_roots += int(int(row["unknown_token_kind_count"]) > 0)
    stats.context_missing_roots += int(bool(row["context_missing_fields"]))
    stats.max_logit_abs_error = max(
        stats.max_logit_abs_error,
        float(row["max_logit_abs_error"]),
    )
    stats.max_value_abs_error = max(
        stats.max_value_abs_error,
        float(row["value_abs_error"]),
    )
    stats.max_prior_kl = max(stats.max_prior_kl, float(row["complete_action_prior_kl"]))
    stats.min_top_k_recall = min(stats.min_top_k_recall, float(row["root_top_k_recall"]))


def _parity_summary(
    config: SearchFeatureParityConfig,
    policy: CheckpointPolicy,
    stats: _ParityStats,
    output_path: Path,
) -> dict[str, Any]:
    roots = stats.roots
    checks = {
        "root_count": roots >= config.min_roots,
        "state_option_bitwise": stats.bitwise_matches == roots,
        "root_top1": stats.top1_matches == roots,
        "root_top_k_recall": stats.min_top_k_recall >= 1.0,
        "logit_error": stats.max_logit_abs_error <= config.logit_atol,
        "prior_kl": stats.max_prior_kl <= config.prior_kl_atol,
        "value_error": stats.max_value_abs_error <= config.value_atol,
        "context_complete": stats.context_missing_roots == 0,
        "hidden_permutation": stats.hidden_matches == roots,
        "unknown_token_drift": stats.unknown_drift_roots == 0,
    }
    return {
        "protocol": "ITS-P0-FEATURE-PARITY-v1",
        "roots": roots,
        "replays": stats.replays,
        "phase_counts": dict(sorted(stats.phase_counts.items())),
        "bitwise_match_rate": _safe_rate(stats.bitwise_matches, roots),
        "top1_match_rate": _safe_rate(stats.top1_matches, roots),
        "hidden_permutation_match_rate": _safe_rate(stats.hidden_matches, roots),
        "min_top_k_recall": stats.min_top_k_recall if roots else 0.0,
        "max_logit_abs_error": stats.max_logit_abs_error,
        "max_complete_action_prior_kl": stats.max_prior_kl,
        "max_value_abs_error": stats.max_value_abs_error,
        "context_missing_roots": stats.context_missing_roots,
        "unknown_drift_roots": stats.unknown_drift_roots,
        "decision_role": "diagnostic_only",
        "checks": checks,
        "diagnostic_warnings": [
            name for name, observed in checks.items() if not observed
        ],
        "source_fingerprint": stats.source_hash.hexdigest(),
        "checkpoint_path": str(config.checkpoint_path),
        "policy_device": policy.device,
        "roots_path": records.display_path(output_path),
        "config": config.model_dump(mode="json"),
    }


def _parity_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("episode_id", pa.int64()),
            pa.field("step_index", pa.int32()),
            pa.field("seat", pa.int8()),
            pa.field("phase", pa.string()),
            pa.field("turn", pa.int16()),
            pa.field("select_context", pa.int16()),
            pa.field("option_count", pa.int16()),
            pa.field("candidate_action_count", pa.int16()),
            pa.field("token_count", pa.int16()),
            pa.field("token_kind_ids", pa.list_(pa.int16())),
            pa.field("token_kind_counts", pa.list_(pa.int16())),
            pa.field("state_option_bitwise_match", pa.bool_()),
            pa.field("hidden_permutation_match", pa.bool_()),
            pa.field("root_top1_match", pa.bool_()),
            pa.field("root_top_k_recall", pa.float64()),
            pa.field("max_logit_abs_error", pa.float64()),
            pa.field("complete_action_prior_kl", pa.float64()),
            pa.field("serving_value", pa.float64()),
            pa.field("adapter_value", pa.float64()),
            pa.field("value_abs_error", pa.float64()),
            pa.field("context_missing_fields", pa.list_(pa.string())),
            pa.field("unknown_token_kind_count", pa.int16()),
            pa.field("probe_feature_count", pa.int16()),
        ]
    )


def _top_actions(
    priors: Mapping[tuple[int, ...], float],
    top_k: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        action
        for action, _ in sorted(
            priors.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    )


def _prior_kl(
    left: Mapping[tuple[int, ...], float],
    right: Mapping[tuple[int, ...], float],
) -> float:
    epsilon = 1.0e-12
    return sum(
        probability
        * math.log(max(probability, epsilon) / max(right.get(action, 0.0), epsilon))
        for action, probability in left.items()
        if probability > 0.0
    )


def _max_abs_error(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    if not left:
        return 0.0
    errors = [
        0.0
        if left_value == right_value
        else abs(float(left_value) - float(right_value))
        for left_value, right_value in zip(left, right, strict=True)
    ]
    return max(errors)


def _phase(observation: Mapping[str, Any], config: SearchFeatureParityConfig) -> str:
    turn = _int_field(observation.get("current"), "turn", -1)
    if turn <= config.early_turn_max:
        return "early"
    if turn <= config.mid_turn_max:
        return "mid"
    return "late"


def _is_own_decision(observation: Mapping[str, Any], seat: int) -> bool:
    return (
        isinstance(observation.get("select"), Mapping)
        and _int_field(observation.get("current"), "yourIndex", -1) == seat
    )


def _resolve_belief_config(
    config: OpponentBeliefFeatureConfig,
) -> OpponentBeliefFeatureConfig:
    if config.deck_signature_summary_path is None:
        return config
    return config.model_copy(
        update={
            "deck_signature_summary_path": records.repo_path(
                config.deck_signature_summary_path
            )
        }
    )


def _resolve_sampler_config(config: BeliefSamplerConfig) -> BeliefSamplerConfig:
    if config.prior_deck_signature_summary_path is None:
        return config
    return config.model_copy(
        update={
            "prior_deck_signature_summary_path": records.repo_path(
                config.prior_deck_signature_summary_path
            )
        }
    )


def _resolve_replay_paths(config: SearchFeatureParityConfig) -> tuple[Path, ...]:
    if config.replay_paths:
        paths = tuple(records.repo_path(path) for path in config.replay_paths)
    else:
        paths = tuple(
            Path(path)
            for path in sorted(glob.glob(str(records.repo_path(Path(config.replay_glob)))))
        )
    if config.max_replays is not None:
        paths = paths[: config.max_replays]
    if not paths:
        raise ValueError("no replay paths matched feature parity audit")
    return paths


def _resolve_seat(metadata: Mapping[str, Any], config: SearchFeatureParityConfig) -> int:
    if config.seat_index is not None:
        return config.seat_index
    team_names = _sequence(_mapping(metadata.get("info")).get("TeamNames"))
    matches = [
        index
        for index, team_name in enumerate(team_names)
        if str(team_name) == str(config.team_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one team_name={config.team_name!r} seat, found {matches}"
        )
    return matches[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / float(denominator) if denominator > 0 else 0.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    if isinstance(value, Mapping):
        item = value.get(name, default)
    else:
        item = getattr(value, name, default)
    return int(item) if item is not None else default
