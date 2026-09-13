"""Artifact-bound whole-match action parity for the native league adapter."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.evaluation.continuous_league.discovery import inspect_checkpoint_pair
from ptcg_rl.evaluation.continuous_league.models import (
    BundleIdentity,
    ContinuousLeagueConfig,
    ControllerKind,
    MatchLease,
)
from ptcg_rl.evaluation.continuous_league.native_match import (
    NativeMatchExecutor,
    match_seeds,
    native_match_contract_fingerprints,
)
from ptcg_rl.evaluation.continuous_league.native_runtime import (
    ControllerFactory,
    MatchAgent,
)
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.training.source_identity import resolve_training_source_identity


class _ActionAuditor:
    """Independent Python ActTime agents consuming the exact served observation."""

    def __init__(self, agents: tuple[MatchAgent, MatchAgent]) -> None:
        self.agents = agents
        self.comparisons = [0, 0]
        self.observation_hashes: list[str] = []

    def __call__(
        self,
        seat: int,
        observation: Mapping[str, Any],
        served_action: tuple[int, ...],
    ) -> None:
        """Require the independent oracle to choose the identical legal action."""
        oracle_observation = copy.deepcopy(dict(observation))
        oracle_action = tuple(
            int(index) for index in self.agents[seat].act(oracle_observation)
        )
        if not is_legal_action(oracle_observation.get("select"), oracle_action):
            raise RuntimeError(
                f"Python ActTime oracle emitted an illegal action at seat {seat}"
            )
        if oracle_action != served_action:
            raise RuntimeError(
                "native adapter/Python ActTime action parity mismatch at "
                f"seat {seat}, decision {self.comparisons[seat]}: "
                f"served={served_action}, oracle={oracle_action}"
            )
        self.comparisons[seat] += 1
        self.observation_hashes.append(_observation_fingerprint(oracle_observation))


def run_checkpoint_action_parity(
    config: ContinuousLeagueConfig,
    *,
    repo_root: Path,
    checkpoint_pair_path: Path,
    deck_path: Path,
    opponent_script: str,
    candidate_seat: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run and immutably publish one whole-match Python action certificate."""
    if candidate_seat not in (0, 1):
        raise ValueError("candidate_seat must be 0 or 1")
    root = repo_root.resolve()
    source_commit = resolve_training_source_identity(root).source_git_commit
    runtime_fingerprint, belief_fingerprint = native_match_contract_fingerprints(
        config.native_match
    )
    if (
        runtime_fingerprint != config.runtime_fingerprint
        or belief_fingerprint != config.belief_fingerprint
    ):
        raise ValueError("native parity profile has stale runtime/belief fingerprints")
    pair_path = _resolve_file(checkpoint_pair_path, root)
    exact_deck_path = _resolve_file(deck_path, root)
    pair = inspect_checkpoint_pair(pair_path)
    deck = canonicalize_deck(records.read_deck(exact_deck_path))
    if deck.deck_digest not in pair.active_deck_digests:
        raise ValueError("parity deck is not an active exact route in the checkpoint")
    script_name = opponent_script.strip()
    if not script_name:
        raise ValueError("opponent_script must be non-empty")
    match_id = _match_id(
        pair.policy_sha256,
        deck.deck_digest,
        script_name,
        candidate_seat,
        config.runtime_fingerprint,
        config.belief_fingerprint,
    )
    checkpoint_bundle = BundleIdentity(
        bundle_id=f"parity-checkpoint-{pair.policy_sha256[:24]}",
        controller_id=f"checkpoint:{pair.policy_sha256}",
        deck_digest=deck.deck_digest,
    )
    script_bundle = BundleIdentity(
        bundle_id=f"parity-script-{hashlib.sha256(script_name.encode()).hexdigest()[:24]}",
        controller_id=f"script:{script_name}",
        deck_digest=deck.deck_digest,
    )
    sides: tuple[
        tuple[BundleIdentity, ControllerKind, Path | None],
        tuple[BundleIdentity, ControllerKind, Path | None],
    ] = (
        (checkpoint_bundle, "checkpoint", pair.policy_path)
        if candidate_seat == 0
        else (script_bundle, "script", None),
        (checkpoint_bundle, "checkpoint", pair.policy_path)
        if candidate_seat == 1
        else (script_bundle, "script", None),
    )
    lease = MatchLease(
        match_id=match_id,
        side_a=sides[0][0],
        side_b=sides[1][0],
        side_a_controller_kind=sides[0][1],
        side_b_controller_kind=sides[1][1],
        side_a_controller_path=sides[0][2],
        side_b_controller_path=sides[1][2],
        side_a_deck_path=exact_deck_path,
        side_b_deck_path=exact_deck_path,
        requires_cuda=True,
        runtime_fingerprint=config.runtime_fingerprint,
        belief_fingerprint=config.belief_fingerprint,
        lease_expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    _, agent_seeds = match_seeds(match_id)
    oracle_factory = ControllerFactory(config.native_match, repo_root=root)
    oracle_agents: list[MatchAgent] = []
    executor = NativeMatchExecutor(config.native_match, repo_root=root)
    try:
        for seat, (bundle, kind, controller_path) in enumerate(sides):
            oracle_agents.append(
                oracle_factory.build(
                    bundle=bundle,
                    kind=kind,
                    controller_path=controller_path,
                    deck_path=exact_deck_path,
                    deck=deck.card_ids,
                    seat=seat,
                    seed=agent_seeds[seat],
                )
            )
        auditor = _ActionAuditor((oracle_agents[0], oracle_agents[1]))
        response = executor.execute(lease, action_auditor=auditor)
        if response.terminal_reason == "infrastructure_error":
            raise RuntimeError(
                f"native parity match failed: {response.telemetry.get('error')}"
            )
        certificate = {
            "format": "continuous_league_native_act_time_parity_v1",
            "passed": True,
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": source_commit,
            "checkpoint_pair_path": records.display_path(pair.manifest_path),
            "checkpoint_pair_sha256": pair.manifest_sha256,
            "checkpoint_policy_path": records.display_path(pair.policy_path),
            "checkpoint_policy_sha256": pair.policy_sha256,
            "deck_path": records.display_path(exact_deck_path),
            "deck_digest": deck.deck_digest,
            "runtime_fingerprint": config.runtime_fingerprint,
            "belief_fingerprint": config.belief_fingerprint,
            "native_library_sha256": executor.library_sha256,
            "match_id": match_id,
            "candidate_seat": candidate_seat,
            "opponent_script": script_name,
            "outcome": response.outcome,
            "terminal_reason": response.terminal_reason,
            "engine_steps": response.steps,
            "action_comparisons": auditor.comparisons,
            "observation_trace_sha256": _trace_fingerprint(auditor.observation_hashes),
        }
        resolved_output = (
            output_path if output_path.is_absolute() else root / output_path
        ).resolve()
        atomic_write_bytes(
            resolved_output,
            json_payload(certificate),
            overwrite=False,
        )
        return certificate
    finally:
        for agent in oracle_agents:
            with suppress(Exception):
                agent.close()
        oracle_factory.close()
        executor.close()


def _match_id(*parts: object) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"parity-{hashlib.sha256(encoded).hexdigest()[:32]}"


def _observation_fingerprint(observation: Mapping[str, Any]) -> str:
    import json

    encoded = json.dumps(
        observation,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trace_fingerprint(hashes: list[str]) -> str:
    digest = hashlib.sha256(b"ptcg-rl/continuous-league/parity-trace/v1\0")
    for value in hashes:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _resolve_file(path: Path, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


__all__ = ["run_checkpoint_action_parity"]
