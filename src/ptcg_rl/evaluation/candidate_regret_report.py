"""Privacy-safe rows and summary for candidate-regret evidence."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ptcg_rl.agent.search.candidate_budget import CandidateBudgetPlan
from ptcg_rl.agent.search.candidates import CandidateConstructionResult
from ptcg_rl.agent.search.hierarchical_contract import (
    DeploymentContinuationControllerIdentity,
)
from ptcg_rl.engine.consequence_identity import (
    candidate_action_fingerprint,
    root_token_fingerprint,
)
from ptcg_rl.evaluation.candidate_regret_config import CandidateRegretAuditConfig
from ptcg_rl.evaluation.candidate_regret_corpus import CandidateAuditCase
from ptcg_rl.evaluation.candidate_regret_engine import (
    ExactCandidateScores,
    UnsupportedExactReference,
)
from ptcg_rl.evaluation.candidate_regret_sources import CandidateSourceTelemetry


class CandidateAuditStats:
    """Streaming diagnostic aggregates; none are quality gates."""

    def __init__(self, k_values: Sequence[int]) -> None:
        self.status_counts: Counter[str] = Counter()
        self.failure_counts: Counter[str] = Counter()
        self.regret_sum = dict.fromkeys(k_values, 0.0)
        self.recall_count = dict.fromkeys(k_values, 0)
        self.regret_count = dict.fromkeys(k_values, 0)
        self.near_best_provenance: Counter[str] = Counter()
        self.exact_roots = 0
        self.exact_decks: set[str] = set()
        self.exact_contexts: set[int] = set()
        self.candidate_rows = 0
        self.regret_rows = 0
        self.native_call_ms = 0.0
        self.scenario_grid_complete_roots = 0
        self.paired_support_integrity_roots = 0
        self.nonanticipativity_integrity_roots = 0


def build_root_row(
    case: CandidateAuditCase,
    *,
    config: CandidateRegretAuditConfig,
    budget: CandidateBudgetPlan,
    result: CandidateConstructionResult,
    telemetry: CandidateSourceTelemetry,
    exact: ExactCandidateScores | UnsupportedExactReference,
    alias_groups: int,
    alias_disagreements: int,
) -> dict[str, Any]:
    """Build one compact root record without raw states or card identities."""
    retained = result.candidates.actions
    if isinstance(exact, ExactCandidateScores):
        status = "comparable_exhaustive_reference"
        support_fingerprint: str | None = exact.scenario_support_fingerprint
        post_state_diversity = len(set(exact.successor_fingerprints.values()))
        leaf_bootstrapped = exact.leaf_bootstrapped
        unique_value_rows = exact.unique_value_rows
        native_errors: Mapping[str, int] = {}
        scorer_fingerprint: str | None = exact.scorer_fingerprint
        controller_fingerprint: str | None = exact.controller_fingerprint
    else:
        status = exact.status
        support_fingerprint = exact.scenario_support_fingerprint
        post_state_diversity = 0
        leaf_bootstrapped = False
        unique_value_rows = 0
        native_errors = exact.native_error_counts
        scorer_fingerprint = None
        controller_fingerprint = None
    select = case.observation.get("select")
    return {
        "case_id": case.case_id,
        "root_state_fingerprint": root_token_fingerprint(
            case.consequence.state_token
        ),
        "deck_fingerprint": case.deck_fingerprint,
        "stratum_fingerprint": case.stratum_fingerprint,
        "select_type": case.select_type,
        "select_context": int(mapping_field(select, "context", 0)),
        "option_count": len(options(select)),
        "min_count": int(mapping_field(select, "minCount", 0)),
        "max_count": int(mapping_field(select, "maxCount", 0)),
        "ordered": case.ordered,
        "legal_action_count": case.legal_action_count,
        "reference_exhaustive": True,
        "rules_exact": exact.rules_exact,
        "status": status,
        "scenario_count_requested": config.engine.scenario_count,
        "scenario_count_used": exact.scenario_count,
        "scenario_support_mode": exact.scenario_support_mode,
        "scenario_support_exhaustive": False,
        "scenario_support_fingerprint": support_fingerprint,
        "scenario_grid_complete": exact.scenario_grid_complete,
        "paired_support_integrity": exact.paired_support_integrity,
        "nonanticipativity_integrity": exact.nonanticipativity_integrity,
        "constructor_valid": result.valid,
        "constructor_fallback_reason": result.fallback_reason,
        "constructor_exhaustive_branch": budget.exhaustive,
        "retained_action_support_complete": (
            len(retained) == case.legal_action_count
        ),
        "seed_count": result.candidates.seed_count,
        "retained_count": len(retained),
        "offered_by_source_json": _json(telemetry.offered_by_source),
        "configured_seed_quotas_json": _json(telemetry.configured_seed_quotas),
        "used_seed_quotas_json": _json(telemetry.used_seed_quotas),
        "configured_expansion_quotas_json": _json(
            telemetry.configured_expansion_quotas
        ),
        "used_expansion_quotas_json": _json(telemetry.used_expansion_quotas),
        "provenance_counts_json": _json(telemetry.provenance_counts),
        "refill_slots": telemetry.refill_slots,
        "duplicate_offers": telemetry.duplicate_offers,
        "multi_source_candidates": telemetry.multi_source_candidates,
        "retained_cardinality_count": len({len(action) for action in retained}),
        "post_state_diversity": post_state_diversity,
        "post_state_alias_groups": alias_groups,
        "post_state_alias_score_disagreements": alias_disagreements,
        "leaf_bootstrapped": leaf_bootstrapped,
        "unique_endpoint_value_rows": unique_value_rows,
        "endpoint_counts_json": _json(exact.endpoint_counts),
        "native_error_counts_json": _json(native_errors),
        "scorer_fingerprint": scorer_fingerprint,
        "controller_fingerprint": controller_fingerprint,
        "native_pack_ms": exact.native_pack_ms,
        "native_call_ms": exact.native_call_ms,
        "native_parse_ms": exact.native_parse_ms,
        "native_payload_bytes": exact.native_payload_bytes,
    }


def build_candidate_rows(
    case: CandidateAuditCase,
    *,
    exact: ExactCandidateScores,
    actions: Sequence[tuple[int, ...]],
    result: CandidateConstructionResult,
    epsilon: float,
) -> tuple[dict[str, Any], ...]:
    """Build exhaustive candidate scores with retained provenance."""
    retained_rank = {
        action: index + 1 for index, action in enumerate(result.candidates.actions)
    }
    sources = dict(
        zip(result.candidates.actions, result.candidates.sources, strict=True)
    )
    best = max(exact.scores.values())
    rows: list[dict[str, Any]] = []
    for action in actions:
        aggregate = exact.aggregates[action]
        rows.append(
            {
                "case_id": case.case_id,
                "candidate_fingerprint": candidate_action_fingerprint(action),
                "action_length": len(action),
                "robust_score": aggregate.robust_score,
                "weighted_mean": aggregate.weighted_mean,
                "weighted_std": aggregate.weighted_std,
                "downside_minimum": aggregate.downside_minimum,
                "terminal_weight": aggregate.terminal_weight,
                "same_seat_main_weight": aggregate.same_seat_main_weight,
                "turn_handoff_weight": aggregate.turn_handoff_weight,
                "information_history_forked": aggregate.information_history_forked,
                "retained_rank": retained_rank.get(action),
                "sources": list(sources.get(action, ())),
                "near_best": exact.scores[action] >= best - epsilon,
                "root_observable_successor_fingerprint": (
                    exact.successor_fingerprints[action]
                ),
            }
        )
    return tuple(rows)


def build_summary(
    config: CandidateRegretAuditConfig,
    *,
    sampling: Mapping[str, Any],
    stats: CandidateAuditStats,
    elapsed_seconds: float,
    checkpoint_sha256: str,
    library_sha256: str,
    native_abi_fingerprint: str,
    constructor_fingerprint: str,
    resolved_fingerprint: str,
    controller: DeploymentContinuationControllerIdentity,
    part_counts: Mapping[str, int],
    stage_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Build the diagnostic-only campaign summary and applicability boundary."""
    metric_curve = {
        str(k): {
            "roots": stats.regret_count[k],
            "mean_best_regret": (
                stats.regret_sum[k] / stats.regret_count[k]
                if stats.regret_count[k]
                else None
            ),
            "epsilon_recall_rate": (
                stats.recall_count[k] / stats.regret_count[k]
                if stats.regret_count[k]
                else None
            ),
        }
        for k in config.k_values
    }
    return {
        "schema": {"name": "candidate_regret_audit", "version": 1},
        "purpose": "diagnostic evidence only; no quality gate or promotion action",
        "identity": _identity_summary(
            config,
            checkpoint_sha256=checkpoint_sha256,
            library_sha256=library_sha256,
            native_abi_fingerprint=native_abi_fingerprint,
            constructor_fingerprint=constructor_fingerprint,
            resolved_fingerprint=resolved_fingerprint,
            controller=controller,
        ),
        "execution": {
            "policy_value_device": config.device,
            "native_device": "cpu",
            "native_cpu_reason": (
                "the bundled simulator is engine-bound and exposes no GPU path"
            ),
            "elapsed_seconds": elapsed_seconds,
            "stage_seconds": dict(stage_seconds),
            "scan_rows_per_second": (
                float(sampling["scanned_rows"]) / stage_seconds["sampling"]
                if stage_seconds["sampling"] > 0.0
                else 0.0
            ),
            "native_call_ms": stats.native_call_ms,
            "proposal_source_mode": config.proposal_source_mode,
            "value_adapter_mode": (
                "immutable pre-schema9 checkpoint; zero-initialized root adapter "
                "is numerically equivalent to the checkpoint root critic"
            ),
        },
        "sampling": dict(sampling),
        "applicability": _applicability(config, sampling=sampling, stats=stats),
        "results": {
            "status_counts": dict(sorted(stats.status_counts.items())),
            "failure_counts": dict(sorted(stats.failure_counts.items())),
            "best_regret_epsilon_recall_by_k": metric_curve,
            "near_best_retained_provenance": dict(
                sorted(stats.near_best_provenance.items())
            ),
            "candidate_rows": stats.candidate_rows,
            "regret_rows": stats.regret_rows,
        },
        "integrity": {
            "paired_scenario_support_fixed_before_candidate_scoring": True,
            "candidate_major_common_weights": True,
            "scenario_conditioned_root_action_selection": False,
            "continuation_decisions_present": False,
            "direct_outcome_nonanticipativity_validated_by_shared_aggregator": True,
            "scenario_grid_complete_roots": stats.scenario_grid_complete_roots,
            "paired_support_integrity_roots": stats.paired_support_integrity_roots,
            "nonanticipativity_integrity_roots": (
                stats.nonanticipativity_integrity_roots
            ),
        },
        "limitations": _limitations(stats),
        "artifacts": {
            "format": "streamed atomic Parquet parts",
            "part_counts": dict(part_counts),
        },
    }


def mapping_field(value: Any, name: str, default: Any) -> Any:
    """Read one mapping/object field for reconstructed engine observations."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def options(select: Any) -> Sequence[Any]:
    """Return a prompt option sequence without accepting text as a sequence."""
    value = mapping_field(select, "option", ())
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _identity_summary(
    config: CandidateRegretAuditConfig,
    *,
    checkpoint_sha256: str,
    library_sha256: str,
    native_abi_fingerprint: str,
    constructor_fingerprint: str,
    resolved_fingerprint: str,
    controller: DeploymentContinuationControllerIdentity,
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "checkpoint_sha256": checkpoint_sha256,
        "native_library_sha256": library_sha256,
        "native_abi_fingerprint": native_abi_fingerprint,
        "constructor_fingerprint": constructor_fingerprint,
        "scorer_fingerprint": config.scorer.scorer_fingerprint,
        "resolved_semantics_fingerprint": resolved_fingerprint,
        "continuation_controller_fingerprint": controller.controller_fingerprint,
    }


def _applicability(
    config: CandidateRegretAuditConfig,
    *,
    sampling: Mapping[str, Any],
    stats: CandidateAuditStats,
) -> dict[str, Any]:
    return {
        "sampled_roots": sampling["retained_roots"],
        "comparable_exhaustive_roots": stats.exact_roots,
        "comparable_exact_decks": len(stats.exact_decks),
        "comparable_prompt_contexts": len(stats.exact_contexts),
        "exhaustive_reference_cap": config.sampling.exhaustive_reference_cap,
        "reference_definition": (
            "all legal complete actions within the explicit cap, scored on "
            "one fixed paired support by the shared scorer"
        ),
        "global_game_optimality_claim": False,
        "bounded_support_called_oracle": False,
        "candidate_budget_quality_conclusion": False,
    }


def _limitations(stats: CandidateAuditStats) -> dict[str, str]:
    return {
        "unsupported_rng": (
            "manual-coin or internal RNG surfaces are reported separately and "
            "never receive regret metrics"
        ),
        "unsupported_strategic_continuation": (
            "roots requiring another non-forced prompt are reported separately; "
            "this audit does not invent a continuation oracle"
        ),
        "public_search_reference": (
            "per-root public Search parity is not rerun here; exact-rule trust is "
            "bound to the Stage-A-parity-tested native library fingerprint, "
            "avoiding the known public Search prize-state defect"
        ),
        "root_observable_alias": (
            "post-state diversity fingerprints cover root-observable successor "
            "payloads, endpoints, and exact effects; they are not hidden-state "
            "full-successor fingerprints"
        ),
        "proposal_checkpoint": (
            "the migration checkpoint predates learned proposal residuals, so "
            "proposal source ordering is the specified zero-residual base parity"
        ),
        "candidate_source_ranking": (
            "on exhaustive-fit audit roots, base/proposal source ordering uses "
            "teacher-forced complete-action probabilities, then truncates every "
            "source stream at the resolved seed bound; it does not claim an "
            "online prefix-search latency measurement"
        ),
        "scenario_reference": (
            "belief worlds are a bounded count-preserving sample and chance is "
            "unsupported; scenario support is never labeled exhaustive"
        ),
        "empirical_hidden_particles": (
            "rotated replay hidden zones are engine-only empirical, "
            "count-preserving particles; they are not exposed to candidate "
            "generation and make no belief-quality or calibration claim"
        ),
        "diagnostic_coverage": (
            f"only {stats.exact_roots} sampled roots produced a comparable "
            "exhaustive reference; the reported regret curve is diagnostic "
            "coverage, not a candidate-budget quality conclusion"
        ),
    }


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "CandidateAuditStats",
    "build_candidate_rows",
    "build_root_row",
    "build_summary",
    "mapping_field",
    "options",
]
