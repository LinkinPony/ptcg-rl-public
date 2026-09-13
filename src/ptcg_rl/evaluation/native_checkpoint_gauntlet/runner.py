"""Native cross-checkpoint orchestration with per-checkpoint CUDA batching."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.decks.traceability import authoritative_deck_hash
from ptcg_rl.evaluation.continuous_league.models import BundleIdentity
from ptcg_rl.evaluation.continuous_league.native_match import (
    native_match_contract_fingerprints,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    FORMAT,
    RESULTS_FORMAT,
    CheckpointRosterParticipantConfig,
    NativeCheckpointGauntletConfig,
    PolicyEvaluationBinding,
    ScheduledCrossCheckpointGame,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.native_match_replica import (
    build_native_match_config,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.native_match_workers import (
    run_native_match_worker_group,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.schedule import (
    schedule_cross_checkpoint_games,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.scoring import (
    write_final_artifacts,
)
from ptcg_rl.evaluation.native_deck_elo.models import DeckAsset
from ptcg_rl.evaluation.native_deck_elo.storage import (
    publish_or_validate_manifest,
    resolve_file,
    resolve_path,
    verified_file,
    write_progress,
)


def run_native_checkpoint_gauntlet(
    config: NativeCheckpointGauntletConfig,
) -> dict[str, Any]:
    """Run or resume one bounded cross-checkpoint exact-roster campaign."""
    root = records.repo_path(Path(".")).resolve()
    checkpoint_paths = {
        "candidate": _verify_participant(config.candidate, root=root),
        "baseline": _verify_participant(config.baseline, root=root),
    }
    library_path = verified_file(
        config.native_library_path,
        root=root,
        expected_sha256=config.expected_native_library_sha256,
    )
    candidate_native = build_native_match_config(
        config,
        config.candidate,
        root=root,
        library_path=library_path,
    )
    baseline_native = build_native_match_config(
        config,
        config.baseline,
        root=root,
        library_path=library_path,
    )
    candidate_runtime, candidate_belief = native_match_contract_fingerprints(
        candidate_native
    )
    baseline_runtime, baseline_belief = native_match_contract_fingerprints(
        baseline_native
    )
    runtime_fingerprint = _combined_fingerprint(
        "runtime", candidate_runtime, baseline_runtime
    )
    belief_fingerprint = _combined_fingerprint(
        "belief", candidate_belief, baseline_belief
    )
    candidate_decks = _load_decks(config.candidate, root=root)
    baseline_decks = _load_decks(config.baseline, root=root)
    campaign = _campaign_payload(
        config,
        candidate_decks=candidate_decks,
        baseline_decks=baseline_decks,
        runtime_fingerprint=runtime_fingerprint,
        belief_fingerprint=belief_fingerprint,
    )
    campaign_fingerprint = _fingerprint(campaign)
    games = schedule_cross_checkpoint_games(
        candidate_decks,
        baseline_decks,
        total_games=config.total_games,
        seed=config.seed,
        campaign_fingerprint=campaign_fingerprint,
        match_seed_namespace=config.match_seed_namespace,
    )
    output_dir = resolve_path(config.output_dir, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "games_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    publish_or_validate_manifest(
        output_dir / "manifest.json",
        {**campaign, "campaign_fingerprint": campaign_fingerprint},
    )
    rows = _load_parts(
        parts_dir,
        games=games,
        campaign_fingerprint=campaign_fingerprint,
    )
    completed = {int(row["game_index"]) for row in rows}
    remaining = tuple(game for game in games if game.game_index not in completed)
    started_clock = time.perf_counter()
    resumed_games = len(rows)
    progress_path = output_dir / "progress.json"
    write_progress(
        progress_path,
        total_games=len(games),
        completed_games=len(rows),
        resumed_games=resumed_games,
        started_clock=started_clock,
        complete=not remaining,
    )

    if remaining:
        run_native_match_worker_group(
            config,
            root=root,
            candidate_native=candidate_native,
            baseline_native=baseline_native,
            checkpoint_paths=checkpoint_paths,
            games=remaining,
            rows=rows,
            parts_dir=parts_dir,
            progress_path=progress_path,
            campaign_fingerprint=campaign_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            belief_fingerprint=belief_fingerprint,
            started_clock=started_clock,
            resumed_games=resumed_games,
        )

    rows = _load_parts(
        parts_dir,
        games=games,
        campaign_fingerprint=campaign_fingerprint,
    )
    if len(rows) != len(games):
        raise RuntimeError(
            f"checkpoint gauntlet retained {len(rows)} games, expected {len(games)}"
        )
    summary = write_final_artifacts(
        config,
        rows=rows,
        output_dir=output_dir,
        campaign_fingerprint=campaign_fingerprint,
        elapsed_seconds=time.perf_counter() - started_clock,
        resumed_games=resumed_games,
    )
    write_progress(
        progress_path,
        total_games=len(games),
        completed_games=len(rows),
        resumed_games=resumed_games,
        started_clock=started_clock,
        complete=True,
    )
    return summary


def _verify_participant(
    participant: CheckpointRosterParticipantConfig,
    *,
    root: Path,
) -> Path:
    checkpoint = verified_file(
        participant.checkpoint_path,
        root=root,
        expected_sha256=participant.expected_checkpoint_sha256,
    )
    catalog_path = verified_file(
        participant.public_catalog_manifest_path,
        root=root,
        expected_sha256=participant.expected_public_catalog_manifest_sha256,
    )
    source_payload: Mapping[str, Any] | None = None
    if participant.source_identity_path is not None:
        if participant.expected_source_identity_sha256 is None:
            raise AssertionError("validated source identity SHA is missing")
        source_path = verified_file(
            participant.source_identity_path,
            root=root,
            expected_sha256=participant.expected_source_identity_sha256,
        )
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if source_payload.get("source_git_commit") != (
            participant.checkpoint_source_commit
        ):
            raise ValueError(f"{participant.label} source identity commit differs")
        if (
            participant.provenance_fingerprint is not None
            and source_payload.get("training_source_fingerprint")
            != participant.provenance_fingerprint
        ):
            raise ValueError(f"{participant.label} source provenance differs")
    resolved_payload: Mapping[str, Any] | None = None
    if participant.resolved_config_path is not None:
        if participant.expected_resolved_config_sha256 is None:
            raise AssertionError("validated resolved config SHA is missing")
        resolved_path = verified_file(
            participant.resolved_config_path,
            root=root,
            expected_sha256=participant.expected_resolved_config_sha256,
        )
        resolved_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if participant.pair_manifest_path is not None:
        if participant.expected_pair_manifest_sha256 is None:
            raise AssertionError("validated checkpoint pair SHA is missing")
        pair_path = verified_file(
            participant.pair_manifest_path,
            root=root,
            expected_sha256=participant.expected_pair_manifest_sha256,
        )
        payload = json.loads(pair_path.read_text(encoding="utf-8"))
        policy = payload.get("policy")
        if not isinstance(policy, Mapping) or policy.get("sha256") != (
            participant.expected_checkpoint_sha256
        ):
            raise ValueError(f"{participant.label} pair does not bind the checkpoint")
        identity = payload.get("metadata", {}).get(
            "simple_stateless_identity",
            {},
        )
        active = identity.get("active_exact_deck_digests")
        if not isinstance(active, list) or not all(
            isinstance(item, str) for item in active
        ):
            raise ValueError(f"{participant.label} pair has no exact roster identity")
        _validate_resolved_identity(
            participant.label,
            identity=identity,
            resolved=resolved_payload,
        )
    else:
        binding_path_value = participant.policy_evaluation_binding_path
        binding_sha = participant.expected_policy_evaluation_binding_sha256
        if binding_path_value is None or binding_sha is None:
            raise AssertionError("validated policy evaluation binding is missing")
        binding_path = verified_file(
            binding_path_value,
            root=root,
            expected_sha256=binding_sha,
        )
        binding = PolicyEvaluationBinding.model_validate_json(binding_path.read_bytes())
        if (
            resolve_file(binding.checkpoint_path, root=root) != checkpoint
            or binding.checkpoint_sha256 != participant.expected_checkpoint_sha256
            or binding.checkpoint_size_bytes != checkpoint.stat().st_size
        ):
            raise ValueError(
                f"{participant.label} evaluation binding differs from checkpoint"
            )
        if binding.checkpoint_source_commit != (participant.checkpoint_source_commit):
            raise ValueError(f"{participant.label} evaluation binding source differs")
        if binding.provenance_fingerprint != participant.provenance_fingerprint:
            raise ValueError(
                f"{participant.label} evaluation binding provenance differs"
            )
        if (
            binding.public_catalog_manifest_sha256
            != participant.expected_public_catalog_manifest_sha256
            or resolve_file(binding.public_catalog_manifest_path, root=root)
            != catalog_path
        ):
            raise ValueError(f"{participant.label} evaluation binding catalog differs")
        if participant.source_identity_path is not None and (
            resolve_file(binding.source_identity_path, root=root)
            != resolve_file(participant.source_identity_path, root=root)
            or binding.source_identity_sha256
            != participant.expected_source_identity_sha256
        ):
            raise ValueError(
                f"{participant.label} evaluation binding provenance differs"
            )
        if participant.resolved_config_path is not None and (
            resolve_file(binding.resolved_config_path, root=root)
            != resolve_file(participant.resolved_config_path, root=root)
            or binding.resolved_config_sha256
            != participant.expected_resolved_config_sha256
        ):
            raise ValueError(f"{participant.label} evaluation binding config differs")
        identity = binding.policy_identity.model_dump(mode="json")
        active = list(binding.policy_identity.active_exact_deck_digests)
        _validate_resolved_identity(
            participant.label,
            identity=identity,
            resolved=resolved_payload,
        )
    configured = {
        canonicalize_deck(
            records.read_deck(resolve_file(deck.path, root=root))
        ).deck_digest
        for deck in participant.decks
    }
    _validate_configured_roster(
        participant.label,
        roster_scope=participant.roster_scope,
        configured=configured,
        active=set(active),
    )
    return checkpoint


def _validate_resolved_identity(
    participant_label: str,
    *,
    identity: Mapping[str, Any],
    resolved: Mapping[str, Any] | None,
) -> None:
    if resolved is None:
        return
    expected = identity.get("resolved_config_fingerprint")
    if expected is not None and resolved.get("resolved_config_fingerprint") != expected:
        raise ValueError(f"{participant_label} resolved config identity differs")
    expected_registry = identity.get("exact_registry_fingerprint")
    if (
        expected_registry is not None
        and resolved.get("exact_registry_fingerprint") != expected_registry
    ):
        raise ValueError(f"{participant_label} resolved registry identity differs")


def _validate_configured_roster(
    participant_label: str,
    *,
    roster_scope: str,
    configured: set[str],
    active: set[str],
) -> None:
    """Validate an exact-roster selection against checkpoint-bound identities."""
    extra = sorted(configured - active)
    if extra:
        raise ValueError(
            f"{participant_label} configured roster contains exact decks absent "
            f"from the checkpoint binding: extra={extra}"
        )
    if roster_scope == "full" and configured != active:
        missing = sorted(active - configured)
        raise ValueError(
            f"{participant_label} configured full roster differs from binding: "
            f"missing={missing}"
        )


def _load_decks(
    participant: CheckpointRosterParticipantConfig,
    *,
    root: Path,
) -> tuple[DeckAsset, ...]:
    output: list[DeckAsset] = []
    seen: set[str] = set()
    controller_id = f"checkpoint:{participant.expected_checkpoint_sha256}"
    for item in participant.decks:
        path = resolve_file(item.path, root=root)
        canonical = canonicalize_deck(records.read_deck(path))
        if canonical.deck_digest in seen:
            raise ValueError(f"duplicate {participant.label} exact deck")
        seen.add(canonical.deck_digest)
        deck_hash = authoritative_deck_hash(
            item.label,
            explicit=item.deck_hash,
        )
        output.append(
            DeckAsset(
                label=item.label,
                path=path,
                deck_digest=canonical.deck_digest,
                deck_hash=deck_hash,
                deck_signature=canonical.signature,
                bundle=BundleIdentity(
                    bundle_id=(
                        f"{participant.label}:"
                        f"{participant.expected_checkpoint_sha256[:12]}:{deck_hash}"
                    ),
                    controller_id=controller_id,
                    deck_digest=canonical.deck_digest,
                ),
            )
        )
    return tuple(output)


def _campaign_payload(
    config: NativeCheckpointGauntletConfig,
    *,
    candidate_decks: Sequence[DeckAsset],
    baseline_decks: Sequence[DeckAsset],
    runtime_fingerprint: str,
    belief_fingerprint: str,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "candidate": _participant_payload(config.candidate, candidate_decks),
        "baseline": _participant_payload(config.baseline, baseline_decks),
        "runner_source_commit": config.runner_source_commit,
        "native_library": {
            "path": records.display_path(config.native_library_path),
            "sha256": config.expected_native_library_sha256,
        },
        "runtime_fingerprint": runtime_fingerprint,
        "belief_fingerprint": belief_fingerprint,
        "schedule": {
            "total_games": config.total_games,
            "seed": config.seed,
            "match_seed_namespace": config.match_seed_namespace,
            "cross_roster": True,
            "mirrored_seats": True,
            "full_cell_coverage": True,
        },
        "execution": {
            "backend": config.backend,
            **(
                {"candidate_intervention": config.candidate_intervention}
                if config.candidate_intervention != "baseline"
                else {}
            ),
            "concurrency": config.concurrency,
            "native_match_total_concurrency": (
                config.concurrency * config.native_worker_replicas
                if config.backend == "native_match"
                else None
            ),
            "policy_batch_max_rows": config.policy_batch_max_rows,
            "policy_batch_wait_ms": config.policy_batch_wait_ms,
            "policy_batch_coalesce_temperatures": (
                config.policy_batch_coalesce_temperatures
            ),
            "checkpoint_resident_precision": (config.checkpoint_resident_precision),
            "checkpoint_rollout_inductor": config.checkpoint_rollout_inductor,
            "evaluation_action_only": config.evaluation_action_only,
            "native_lane_worker_count": config.native_lane_worker_count,
            "native_worker_replicas": config.native_worker_replicas,
            "native_worker_torch_threads": config.native_worker_torch_threads,
            **(
                {
                    "native_inductor_compile_threads": (
                        config.native_inductor_compile_threads
                    )
                }
                if config.native_inductor_compile_threads is not None
                else {}
            ),
            "native_option_capacity": config.native_option_capacity,
            "native_arena_capacity": config.native_arena_capacity,
            "native_engine_shards": config.native_engine_shards,
            "native_engine_fact_workers": config.native_engine_fact_workers,
            "native_policy_cohort_slots": config.native_policy_cohort_slots,
            "native_policy_group_bank_limit": (config.native_policy_group_bank_limit),
            "native_policy_cohort_wait_ms": config.native_policy_cohort_wait_ms,
            "native_frozen_batch_min_rows": config.native_frozen_batch_min_rows,
            "native_frozen_batch_max_wait_waves": (
                config.native_frozen_batch_max_wait_waves
            ),
            "collection_part_games": config.collection_part_games,
            "maximum_engine_steps": config.maximum_engine_steps,
            "device": config.device,
            "act_time": config.act_time.model_dump(mode="json"),
        },
    }


def _participant_payload(
    participant: CheckpointRosterParticipantConfig,
    decks: Sequence[DeckAsset],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": participant.label,
        "checkpoint_path": records.display_path(participant.checkpoint_path),
        "checkpoint_sha256": participant.expected_checkpoint_sha256,
        "checkpoint_source_commit": participant.checkpoint_source_commit,
        "public_catalog_manifest_path": records.display_path(
            participant.public_catalog_manifest_path
        ),
        "public_catalog_manifest_sha256": (
            participant.expected_public_catalog_manifest_sha256
        ),
        "provenance_fingerprint": participant.provenance_fingerprint,
        "policy_temperature": participant.policy_temperature,
        "roster_scope": participant.roster_scope,
        "decks": [
            {
                "label": deck.label,
                "path": records.display_path(deck.path),
                "deck_digest": deck.deck_digest,
                "deck_hash": deck.deck_hash,
                "deck_signature": deck.deck_signature,
            }
            for deck in decks
        ],
    }
    if participant.source_identity_path is not None:
        payload.update(
            source_identity_path=records.display_path(participant.source_identity_path),
            source_identity_sha256=participant.expected_source_identity_sha256,
        )
    if participant.resolved_config_path is not None:
        payload.update(
            resolved_config_path=records.display_path(participant.resolved_config_path),
            resolved_config_sha256=participant.expected_resolved_config_sha256,
        )
    if participant.pair_manifest_path is not None:
        payload.update(
            pair_manifest_path=records.display_path(participant.pair_manifest_path),
            pair_manifest_sha256=participant.expected_pair_manifest_sha256,
        )
    else:
        if participant.policy_evaluation_binding_path is None:
            raise AssertionError("validated evaluation binding path is missing")
        payload.update(
            policy_evaluation_binding_path=records.display_path(
                participant.policy_evaluation_binding_path
            ),
            policy_evaluation_binding_sha256=(
                participant.expected_policy_evaluation_binding_sha256
            ),
        )
    return payload


def _load_parts(
    parts_dir: Path,
    *,
    games: Sequence[ScheduledCrossCheckpointGame],
    campaign_fingerprint: str,
) -> list[dict[str, Any]]:
    expected = {game.game_index: game for game in games}
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted(parts_dir.glob("part-*.parquet")):
        for raw_row in pq.read_table(path).to_pylist():
            row = cast(dict[str, Any], raw_row)
            game_index = int(row["game_index"])
            game = expected.get(game_index)
            if game is None or game_index in seen:
                raise ValueError(f"invalid or duplicate game index in {path}")
            if (
                row.get("format") != RESULTS_FORMAT
                or row.get("campaign_fingerprint") != campaign_fingerprint
                or row.get("match_id") != game.match_id
            ):
                raise ValueError(f"game result identity differs in {path}")
            seen.add(game_index)
            rows.append(row)
    rows.sort(key=lambda row: int(row["game_index"]))
    return rows


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/native-checkpoint-gauntlet/v1\0" + encoded
    ).hexdigest()


def _combined_fingerprint(kind: str, candidate: str, baseline: str) -> str:
    return hashlib.sha256(
        f"ptcg-rl/checkpoint-gauntlet/{kind}/v1\0{candidate}\0{baseline}".encode()
    ).hexdigest()


__all__ = ["run_native_checkpoint_gauntlet"]
