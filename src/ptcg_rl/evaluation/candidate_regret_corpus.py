"""Second-pass loading for selected candidate-regret replay roots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.evaluation.candidate_regret_sampling import CandidateAuditLocator
from ptcg_rl.evaluation.consequence_audit_corpus import (
    ConsequenceAuditCase,
    load_audit_cases,
)
from ptcg_rl.training.bc_dataset import observation_from_step_row


@dataclass(frozen=True, slots=True)
class CandidateAuditCase:
    """One retained exact-engine root with its deployable model observation."""

    consequence: ConsequenceAuditCase
    observation: Mapping[str, Any]
    deck_signature: str
    deck_fingerprint: str
    select_type: int
    legal_action_count: int
    ordered: bool
    stratum_fingerprint: str
    prompt_fingerprint: str

    @property
    def case_id(self) -> str:
        """Return the stable privacy-safe replay-root identity."""
        return self.consequence.case_id


def load_candidate_audit_cases(
    locators: Sequence[CandidateAuditLocator],
    *,
    batch_size: int,
    fallback_card_id: int,
    fallback_basic_pokemon_id: int,
) -> tuple[CandidateAuditCase, ...]:
    """Load hidden material and only selected Parquet row groups."""
    base_cases = load_audit_cases(
        tuple(item.case for item in locators),
        batch_size=batch_size,
        fallback_card_id=fallback_card_id,
        fallback_basic_pokemon_id=fallback_basic_pokemon_id,
    )
    base_by_id = {item.case_id: item for item in base_cases}
    locator_by_id = {item.case_id: item for item in locators}
    observations = _load_selected_observations(locators)
    cases: list[CandidateAuditCase] = []
    for case_id in sorted(locator_by_id):
        locator = locator_by_id[case_id]
        base = base_by_id.get(case_id)
        observation = observations.get(case_id)
        if base is None or observation is None:
            raise ValueError("retained candidate audit row could not be reloaded")
        current = observation.get("current")
        if not isinstance(current, Mapping) or int(current.get("yourIndex", -1)) != (
            base.player_index
        ):
            raise ValueError("reconstructed observation has the wrong root player")
        cases.append(
            CandidateAuditCase(
                consequence=base,
                observation=observation,
                deck_signature=locator.deck_signature,
                deck_fingerprint=locator.deck_fingerprint,
                select_type=locator.select_type,
                legal_action_count=locator.legal_action_count,
                ordered=locator.ordered,
                stratum_fingerprint=locator.stratum_fingerprint,
                prompt_fingerprint=locator.prompt_fingerprint,
            )
        )
    return tuple(cases)


def _load_selected_observations(
    locators: Sequence[CandidateAuditLocator],
) -> dict[str, Mapping[str, Any]]:
    by_path: dict[Path, list[CandidateAuditLocator]] = defaultdict(list)
    for locator in locators:
        by_path[locator.case.path].append(locator)
    result: dict[str, Mapping[str, Any]] = {}
    for path in sorted(by_path):
        parquet_file = pq.ParquetFile(path)
        pending = {item.case.row_index: item for item in by_path[path]}
        absolute_start = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_count = parquet_file.metadata.row_group(row_group_index).num_rows
            selected = sorted(
                index
                for index in pending
                if absolute_start <= index < absolute_start + row_count
            )
            if selected:
                table = parquet_file.read_row_group(row_group_index, use_threads=True)
                local = pa.array(
                    [index - absolute_start for index in selected],
                    type=pa.int64(),
                )
                rows = table.take(local).to_pylist()
                for row_index, raw in zip(selected, rows, strict=True):
                    locator = pending.pop(row_index)
                    row = cast(Mapping[str, Any], raw)
                    if str(row.get("deck_signature") or "") != (locator.deck_signature):
                        raise ValueError(
                            "retained deck signature changed during reload"
                        )
                    result[locator.case_id] = observation_from_step_row(row)
            absolute_start += row_count
        if pending:
            raise ValueError(f"retained row index exceeds Parquet file: {path}")
    return result


__all__ = ["CandidateAuditCase", "load_candidate_audit_cases"]
