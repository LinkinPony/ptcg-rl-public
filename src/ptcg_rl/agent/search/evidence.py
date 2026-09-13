"""Adapters from serving search results to model-safe candidate evidence."""

from __future__ import annotations

import statistics

from ptcg_rl.agent.search.config import PairedRerankConfig
from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.agent.search.reranker import MacroSearchResult, MacroWorldEvaluation
from ptcg_rl.agent.search.scoring import PairedWorldScore, combined_world_score
from ptcg_rl.engine.search_evidence import (
    SearchCandidateEvidence,
    SearchEvidence,
    search_candidate_features,
)

_EVIDENCE_ENDPOINTS = frozenset(
    {
        MacroEndpoint.TERMINAL,
        MacroEndpoint.SAME_SEAT_MAIN,
        MacroEndpoint.TURN_HANDOFF,
    }
)


def search_evidence_from_macro_result(
    result: MacroSearchResult,
    *,
    legal_action_count: int,
    config: PairedRerankConfig,
) -> SearchEvidence | None:
    """Return complete public evidence, or ``None`` for any partial grid.

    Serving search and the asynchronous complete-action teacher use the same
    fixed-width feature constructor.  This adapter deliberately discards the
    retained transitions and all individual leaf observations.
    """
    actions = result.candidates.actions
    if (
        not actions
        or legal_action_count < len(actions)
        or not result.complete_coverage
        or result.stop_reason != "complete"
        or result.worlds_requested <= 0
        or result.worlds_sampled != result.worlds_requested
        or result.worlds_completed != result.worlds_requested
        or result.state_leaks != 0
    ):
        return None

    indexed: dict[tuple[tuple[int, ...], int], MacroWorldEvaluation] = {}
    for row in result.evaluations:
        key = (row.action, row.world_index)
        if key in indexed:
            return None
        indexed[key] = row

    root_coverage = min(
        float(len(actions)) / float(max(1, legal_action_count)),
        1.0,
    )
    candidates: list[SearchCandidateEvidence] = []
    for action in actions:
        rows = []
        scores = []
        for world_index in range(result.worlds_requested):
            item = indexed.get((action, world_index))
            if item is None:
                return None
            row = item
            endpoint = row.endpoint
            if endpoint not in _EVIDENCE_ENDPOINTS or row.error:
                return None
            # The scorer config binds whether serving handoffs retain the
            # engine-only approximation or consume the deployable root adapter.
            score = combined_world_score(
                PairedWorldScore(
                    action=action,
                    world_index=world_index,
                    endpoint=endpoint,
                    engine_score=row.engine_score,
                    critic_value=row.critic_value,
                    error=row.error,
                ),
                config,
            )
            if score is None:
                return None
            rows.append(row)
            scores.append(score)

        mean_score = statistics.fmean(scores)
        score_std = statistics.pstdev(scores)
        world_count = float(len(rows))
        candidates.append(
            SearchCandidateEvidence(
                action=action,
                features=search_candidate_features(
                    world_scores=scores,
                    robust_score=mean_score - config.risk_std_weight * score_std,
                    coverage=root_coverage,
                    terminal_fraction=(
                        sum(row.endpoint == MacroEndpoint.TERMINAL for row in rows)
                        / world_count
                    ),
                    same_seat_main_fraction=(
                        sum(
                            row.endpoint == MacroEndpoint.SAME_SEAT_MAIN for row in rows
                        )
                        / world_count
                    ),
                    turn_handoff_fraction=(
                        sum(row.endpoint == MacroEndpoint.TURN_HANDOFF for row in rows)
                        / world_count
                    ),
                    mean_path_steps=statistics.fmean(float(row.steps) for row in rows),
                    # PairedMacroSearcher follows one continuation policy; it
                    # is never an exhaustive continuation proof.
                    exact=False,
                ),
            )
        )

    return SearchEvidence(
        candidates=tuple(candidates),
        legal_action_count=legal_action_count,
        world_count=result.worlds_requested,
        exhaustive=legal_action_count == len(candidates),
        exact=False,
    )


__all__ = ["search_evidence_from_macro_result"]
