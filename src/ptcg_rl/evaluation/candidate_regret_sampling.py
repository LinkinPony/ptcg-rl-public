"""Bounded cross-deck/prompt sampling for candidate-regret evidence."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.evaluation.candidate_regret_reservoir import (
    BoundedStratifiedReservoir,
    DistinctSketch,
    balanced_union,
)
from ptcg_rl.evaluation.consequence_audit_sampling import CaseLocator, RootShape

_SCAN_COLUMNS = (
    "date",
    "episode_id",
    "step_index",
    "player_index",
    "deck_signature",
    "select_type",
    "select_context",
    "select_min_count",
    "select_max_count",
    "select_option_count",
)
_STRATUM_DOMAIN = b"ptcg-rl/candidate-regret/stratum/v1\x00"
_ROOT_DOMAIN = b"ptcg-rl/candidate-regret/root/v1\x00"
_CORPUS_DOMAIN = b"ptcg-rl/candidate-regret/corpus/v1\x00"


@dataclass(frozen=True, slots=True)
class CandidateAuditLocator:
    """One retained replay row plus privacy-safe deck/prompt stratum identity."""

    case: CaseLocator
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
        return self.case.case_id


@dataclass(frozen=True, slots=True)
class CandidateSamplingReport:
    """Streaming scan coverage and retained cross-stratum composition."""

    scanned_files: int
    scanned_rows: int
    eligible_rows: int
    eligible_strata_estimate: int
    retained_roots: int
    retained_strata: int
    retained_decks: int
    retained_prompt_contexts: int
    legal_action_histogram: Mapping[str, int]
    corpus_fingerprint: str


def sample_candidate_audit_roots(
    paths: Sequence[Path],
    *,
    max_strata: int,
    roots_per_stratum: int,
    max_roots: int,
    min_legal_actions: int,
    exhaustive_reference_cap: int,
    batch_size: int,
    seed: str,
) -> tuple[tuple[CandidateAuditLocator, ...], CandidateSamplingReport]:
    """Stream scalar columns and retain only explicit exhaustive-fit roots."""
    deck_reservoir = BoundedStratifiedReservoir[CandidateAuditLocator](
        max_strata=max_strata,
        roots_per_stratum=roots_per_stratum,
        seed=seed,
    )
    prompt_reservoir = BoundedStratifiedReservoir[CandidateAuditLocator](
        max_strata=max_strata,
        roots_per_stratum=roots_per_stratum,
        seed=f"{seed}/prompt",
    )
    corpus_digest = hashlib.sha256(_CORPUS_DOMAIN)
    scanned_rows = 0
    eligible_rows = 0
    eligible_strata = DistinctSketch()
    legal_counts: Counter[str] = Counter()
    ordered_paths = tuple(sorted(Path(path) for path in paths))
    for path in ordered_paths:
        parquet_file = pq.ParquetFile(path)
        _require_columns(parquet_file.schema_arrow.names, path)
        absolute_row = 0
        for batch in parquet_file.iter_batches(
            columns=list(_SCAN_COLUMNS),
            batch_size=batch_size,
            use_threads=True,
        ):
            for raw in batch.to_pylist():
                row = cast(Mapping[str, Any], raw)
                scanned_rows += 1
                locator = _candidate_locator(
                    path,
                    absolute_row,
                    row,
                    min_legal_actions=min_legal_actions,
                    exhaustive_reference_cap=exhaustive_reference_cap,
                )
                absolute_row += 1
                if locator is None:
                    continue
                eligible_rows += 1
                eligible_strata.add(locator.stratum_fingerprint)
                legal_counts[_legal_count_bucket(locator.legal_action_count)] += 1
                deck_reservoir.add(
                    locator,
                    stratum_key=locator.deck_fingerprint,
                    root_key=locator.case.shape.key,
                    item_key=locator.case_id,
                )
                prompt_reservoir.add(
                    locator,
                    stratum_key=locator.prompt_fingerprint,
                    root_key=locator.case.shape.key,
                    item_key=locator.case_id,
                )
                _update_framed(corpus_digest, locator.case.shape.key.encode("utf-8"))
                _update_framed(
                    corpus_digest,
                    bytes.fromhex(locator.stratum_fingerprint),
                )

    retained = balanced_union(
        deck_reservoir.retained(max_roots=max_roots),
        prompt_reservoir.retained(max_roots=max_roots),
        max_items=max_roots,
        identity=lambda item: item.case_id,
    )
    report = CandidateSamplingReport(
        scanned_files=len(ordered_paths),
        scanned_rows=scanned_rows,
        eligible_rows=eligible_rows,
        eligible_strata_estimate=eligible_strata.estimate,
        retained_roots=len(retained),
        retained_strata=len({item.stratum_fingerprint for item in retained}),
        retained_decks=len({item.deck_fingerprint for item in retained}),
        retained_prompt_contexts=len(
            {item.case.shape.select_context for item in retained}
        ),
        legal_action_histogram=dict(sorted(legal_counts.items())),
        corpus_fingerprint=corpus_digest.hexdigest(),
    )
    return retained, report


def _candidate_locator(
    path: Path,
    row_index: int,
    row: Mapping[str, Any],
    *,
    min_legal_actions: int,
    exhaustive_reference_cap: int,
) -> CandidateAuditLocator | None:
    deck_signature = str(row.get("deck_signature") or "").strip()
    if not deck_signature:
        return None
    option_count = max(0, int(row.get("select_option_count") or 0))
    min_count = min(option_count, max(0, int(row.get("select_min_count") or 0)))
    max_count = min(
        option_count,
        max(min_count, int(row.get("select_max_count") or 0)),
    )
    context = int(row.get("select_context") or 0)
    ordered = context == int(SelectContext.SKILL_ORDER) or max_count > 1
    legal_count = _legal_action_count(
        option_count,
        min_count,
        max_count,
        ordered=ordered,
    )
    if not min_legal_actions <= legal_count <= exhaustive_reference_cap:
        return None
    shape = RootShape(
        date=str(row.get("date") or ""),
        episode_id=int(row.get("episode_id") or 0),
        step_index=int(row.get("step_index") or 0),
        player_index=int(row.get("player_index") or 0),
        select_context=context,
        select_min_count=min_count,
        select_max_count=max_count,
        select_option_count=option_count,
    )
    deck_fingerprint = hashlib.sha256(deck_signature.encode("utf-8")).hexdigest()
    select_type = int(row.get("select_type") or 0)
    stratum = _stratum_fingerprint(
        deck_fingerprint=deck_fingerprint,
        select_type=select_type,
        context=context,
        option_count=option_count,
        min_count=min_count,
        max_count=max_count,
        ordered=ordered,
    )
    prompt_fingerprint = _prompt_fingerprint(
        select_type=select_type,
        context=context,
        option_count=option_count,
        min_count=min_count,
        max_count=max_count,
        ordered=ordered,
    )
    labels = frozenset(
        {
            "ordered" if ordered else "direct",
            "subset" if max_count > 1 else "direct",
        }
    )
    return CandidateAuditLocator(
        case=CaseLocator(
            path=path,
            row_index=row_index,
            shape=shape,
            labels=labels,
        ),
        deck_signature=deck_signature,
        deck_fingerprint=deck_fingerprint,
        select_type=select_type,
        legal_action_count=legal_count,
        ordered=ordered,
        stratum_fingerprint=stratum,
        prompt_fingerprint=prompt_fingerprint,
    )


def _stratum_fingerprint(
    *,
    deck_fingerprint: str,
    select_type: int,
    context: int,
    option_count: int,
    min_count: int,
    max_count: int,
    ordered: bool,
) -> str:
    digest = hashlib.sha256(_STRATUM_DOMAIN)
    digest.update(bytes.fromhex(deck_fingerprint))
    digest.update(
        struct.pack(
            ">6i?",
            select_type,
            context,
            option_count,
            min_count,
            max_count,
            _legal_action_count(
                option_count,
                min_count,
                max_count,
                ordered=ordered,
            ),
            ordered,
        )
    )
    return digest.hexdigest()


def _prompt_fingerprint(
    *,
    select_type: int,
    context: int,
    option_count: int,
    min_count: int,
    max_count: int,
    ordered: bool,
) -> str:
    digest = hashlib.sha256(b"ptcg-rl/candidate-regret/prompt-stratum/v1\x00")
    digest.update(
        struct.pack(
            ">5i?",
            select_type,
            context,
            option_count,
            min_count,
            max_count,
            ordered,
        )
    )
    return digest.hexdigest()


def _legal_action_count(
    option_count: int,
    min_count: int,
    max_count: int,
    *,
    ordered: bool,
) -> int:
    counter = math.perm if ordered else math.comb
    return sum(
        counter(option_count, count) for count in range(min_count, max_count + 1)
    )


def _legal_count_bucket(value: int) -> str:
    lower = 1 << (value.bit_length() - 1)
    upper = (lower << 1) - 1
    return f"{lower}-{upper}"


def _require_columns(names: Sequence[str], path: Path) -> None:
    missing = sorted(set(_SCAN_COLUMNS).difference(names))
    if missing:
        raise ValueError(f"Parquet input {path} is missing columns: {missing}")


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


__all__ = [
    "CandidateAuditLocator",
    "CandidateSamplingReport",
    "sample_candidate_audit_roots",
]
