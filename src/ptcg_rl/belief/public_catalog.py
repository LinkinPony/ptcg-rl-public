"""Immutable complete public-deck catalog and evidence-conditioned posterior."""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.belief.identity import canonical_belief_fingerprint, file_sha256
from ptcg_rl.cards.static_features import DEFAULT_NUM_CARD_IDS
from ptcg_rl.decks.identity import DECK_SIZE, parse_canonical_signature

PUBLIC_DECK_CATALOG_SCHEMA = "public-deck-catalog-v1"
_CATALOG_FINGERPRINT_DOMAIN = b"ptcg-rl/public-deck-catalog/v1\x00"
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class PublicDeckCatalogEntry:
    """One known public exact-deck multiset and its existence support."""

    signature: str
    card_ids: tuple[int, ...]
    counts: tuple[int, ...]
    public_support: int

    def __post_init__(self) -> None:
        """Require canonical sparse counts for exactly 60 cards."""
        if (
            not self.card_ids
            or len(self.card_ids) != len(self.counts)
            or tuple(sorted(self.card_ids)) != self.card_ids
            or len(set(self.card_ids)) != len(self.card_ids)
        ):
            raise ValueError("catalog entry card IDs must be unique and sorted")
        if any(card_id <= 0 for card_id in self.card_ids):
            raise ValueError("catalog entry card IDs must be positive")
        if any(count <= 0 for count in self.counts):
            raise ValueError("catalog entry counts must be positive")
        if sum(self.counts) != DECK_SIZE:
            raise ValueError("catalog entry must contain exactly 60 cards")
        if self.public_support < 0:
            raise ValueError("catalog public support cannot be negative")
        deck = parse_canonical_signature(
            self.signature,
            max_card_id=max(self.card_ids),
        )
        expected = tuple(sorted(Counter(deck.card_ids).items()))
        if expected != tuple(zip(self.card_ids, self.counts, strict=True)):
            raise ValueError("catalog signature differs from sparse card counts")

    @classmethod
    def from_signature(
        cls,
        signature: str,
        *,
        public_support: int,
        card_vocab_size: int = DEFAULT_NUM_CARD_IDS,
    ) -> PublicDeckCatalogEntry:
        """Parse one strict canonical exact-deck signature."""
        deck = parse_canonical_signature(
            signature,
            max_card_id=card_vocab_size,
        )
        sparse = tuple(sorted(Counter(deck.card_ids).items()))
        return cls(
            signature=deck.signature,
            card_ids=tuple(card_id for card_id, _count in sparse),
            counts=tuple(count for _card_id, count in sparse),
            public_support=public_support,
        )


@dataclass(frozen=True, slots=True)
class CatalogPosteriorEntry:
    """Posterior probability for one catalog exact deck."""

    signature: str
    probability: float


@dataclass(frozen=True, slots=True)
class ExpectedRemainingCard:
    """One full-summary expected remaining card count."""

    card_id: int
    expected_count: float


@dataclass(frozen=True, slots=True)
class PublicDeckPosterior:
    """Exact entries plus explicit unknown-tail posterior and raw summary."""

    entries: tuple[CatalogPosteriorEntry, ...]
    unknown_probability: float
    expected_remaining: tuple[ExpectedRemainingCard, ...]
    entropy: float
    compatible_deck_count: int
    public_evidence_count: int


@dataclass(frozen=True, slots=True)
class PublicDeckPosteriorArrays:
    """Compact array-backed model view of a raw public posterior summary."""

    card_ids: npt.NDArray[np.int32]
    expected_counts: npt.NDArray[np.float32]
    unknown_probability: float
    entropy: float
    compatible_deck_count: int
    public_evidence_count: int

    def __post_init__(self) -> None:
        """Require aligned one-dimensional sparse summary arrays."""
        if self.card_ids.ndim != 1 or self.expected_counts.shape != self.card_ids.shape:
            raise ValueError("public posterior arrays must be aligned vectors")

    @classmethod
    def from_posterior(
        cls,
        posterior: PublicDeckPosterior,
    ) -> PublicDeckPosteriorArrays:
        """Drop unused exact-entry diagnostics and pack expected counts."""
        return cls(
            card_ids=np.asarray(
                [item.card_id for item in posterior.expected_remaining],
                dtype=np.int32,
            ),
            expected_counts=np.asarray(
                [item.expected_count for item in posterior.expected_remaining],
                dtype=np.float32,
            ),
            unknown_probability=posterior.unknown_probability,
            entropy=posterior.entropy,
            compatible_deck_count=posterior.compatible_deck_count,
            public_evidence_count=posterior.public_evidence_count,
        )

    @property
    def entries(self) -> tuple[CatalogPosteriorEntry, ...]:
        """Exact-entry probabilities are not consumed by actor or learner."""
        return ()

    @property
    def expected_remaining(self) -> tuple[ExpectedRemainingCard, ...]:
        """Return the compatibility tuple only for persistence/debug callers."""
        return tuple(
            ExpectedRemainingCard(
                card_id=int(card_id),
                expected_count=float(expected_count),
            )
            for card_id, expected_count in zip(
                self.card_ids,
                self.expected_counts,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class NativePublicDeckCatalogArrays:
    """Owned immutable numeric snapshot for a native posterior implementation.

    Card IDs index the second ``entry_counts`` dimension directly, so column
    zero is the unused sentinel. Unknown-card arrays omit that sentinel and
    therefore map card ID ``n`` to index ``n - 1``.
    """

    entry_counts: npt.NDArray[np.int16]
    exact_log_priors: npt.NDArray[np.float64]
    log_combinations: npt.NDArray[np.float64]
    log_factorials: npt.NDArray[np.float64]
    unknown_card_probabilities: npt.NDArray[np.float64]
    unknown_log_card_probabilities: npt.NDArray[np.float64]
    unknown_log_prior: float

    def __post_init__(self) -> None:
        """Require the exact dense, C-contiguous ABI layout."""
        arrays = (
            ("entry_counts", self.entry_counts, np.dtype(np.int16), 2),
            ("exact_log_priors", self.exact_log_priors, np.dtype(np.float64), 1),
            ("log_combinations", self.log_combinations, np.dtype(np.float64), 2),
            ("log_factorials", self.log_factorials, np.dtype(np.float64), 1),
            (
                "unknown_card_probabilities",
                self.unknown_card_probabilities,
                np.dtype(np.float64),
                1,
            ),
            (
                "unknown_log_card_probabilities",
                self.unknown_log_card_probabilities,
                np.dtype(np.float64),
                1,
            ),
        )
        for name, values, dtype, ndim in arrays:
            if (
                values.dtype != dtype
                or values.ndim != ndim
                or not values.flags.c_contiguous
                or values.flags.writeable
            ):
                raise ValueError(
                    f"{name} must be a read-only {dtype} C-contiguous "
                    f"{ndim}-dimensional array"
                )
        entry_count, card_count_with_sentinel = self.entry_counts.shape
        card_count = card_count_with_sentinel - 1
        combination_count = self.log_combinations.shape[0]
        if (
            entry_count <= 0
            or card_count <= 0
            or self.exact_log_priors.shape != (entry_count,)
            or self.log_combinations.shape
            != (combination_count, combination_count)
            or self.log_factorials.shape != (combination_count,)
            or self.unknown_card_probabilities.shape != (card_count,)
            or self.unknown_log_card_probabilities.shape != (card_count,)
        ):
            raise ValueError("native public catalog arrays have inconsistent shapes")
        if not math.isfinite(self.unknown_log_prior):
            raise ValueError("native unknown log prior must be finite")


class PublicDeckCatalogManifest(BaseModel):
    """Small relocatable manifest binding one compact catalog artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_id: str = Field(
        default=PUBLIC_DECK_CATALOG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    artifact_filename: str
    artifact_sha256: str
    catalog_fingerprint: str
    source_snapshot_sha256: str
    card_catalog_fingerprint: str
    card_vocab_size: int
    deck_count: int
    presence_floor: int
    unknown_prior_mass: float

    @field_validator("schema_id")
    @classmethod
    def valid_schema(cls, value: str) -> str:
        """Require the implemented catalog schema."""
        if value != PUBLIC_DECK_CATALOG_SCHEMA:
            raise ValueError("unsupported public deck catalog schema")
        return value

    @field_validator(
        "artifact_sha256",
        "catalog_fingerprint",
        "source_snapshot_sha256",
        "card_catalog_fingerprint",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Normalize immutable lowercase SHA-256 identities."""
        normalized = value.strip().lower()
        if len(normalized) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("catalog fingerprint must contain 64 hex digits")
        return normalized

    @field_validator("artifact_filename")
    @classmethod
    def valid_artifact_filename(cls, value: str) -> str:
        """Keep the manifest relocatable beside a simple artifact filename."""
        if not value or Path(value).name != value:
            raise ValueError("catalog artifact filename must not contain a directory")
        return value

    @field_validator("card_vocab_size", "deck_count", "presence_floor")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Require non-empty catalog dimensions and support floor."""
        if value <= 0:
            raise ValueError("catalog dimensions must be positive")
        return value

    @field_validator("unknown_prior_mass")
    @classmethod
    def valid_unknown_mass(cls, value: float) -> float:
        """Keep the unknown tail explicit and non-degenerate."""
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("unknown_prior_mass must lie strictly in (0, 1)")
        return value


class PublicDeckCatalogConfig(BaseModel):
    """Runtime binding to an immutable public catalog manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: Path
    catalog_fingerprint: str

    @field_validator("catalog_fingerprint")
    @classmethod
    def valid_catalog_fingerprint(cls, value: str) -> str:
        """Require an explicit expected semantic fingerprint."""
        return PublicDeckCatalogManifest.valid_sha256(value)


class PublicDeckCatalog:
    """Complete known public exact decks plus an explicit unknown tail."""

    def __init__(
        self,
        entries: tuple[PublicDeckCatalogEntry, ...],
        *,
        unknown_prior_mass: float,
        presence_floor: int,
        card_vocab_size: int,
        card_catalog_fingerprint: str,
        source_snapshot_sha256: str,
    ) -> None:
        """Validate and precompute support-weighted catalog card marginals."""
        if not entries:
            raise ValueError("public deck catalog must contain at least one deck")
        if not math.isfinite(unknown_prior_mass) or not 0.0 < unknown_prior_mass < 1.0:
            raise ValueError("unknown_prior_mass must lie strictly in (0, 1)")
        if presence_floor <= 0 or card_vocab_size <= 0:
            raise ValueError("presence floor and card vocabulary must be positive")
        _validate_sha256(card_catalog_fingerprint, label="card catalog")
        _validate_sha256(source_snapshot_sha256, label="source snapshot")
        ordered = tuple(sorted(entries, key=lambda entry: entry.signature))
        signatures = tuple(entry.signature for entry in ordered)
        if len(signatures) != len(set(signatures)):
            raise ValueError("public deck catalog contains duplicate exact decks")
        if any(max(entry.card_ids) > card_vocab_size for entry in ordered):
            raise ValueError("catalog card ID exceeds the declared vocabulary")
        self.entries = ordered
        self.unknown_prior_mass = float(unknown_prior_mass)
        self.presence_floor = int(presence_floor)
        self.card_vocab_size = int(card_vocab_size)
        self.card_catalog_fingerprint = card_catalog_fingerprint
        self.source_snapshot_sha256 = source_snapshot_sha256
        self._support_weights = np.asarray(
            [max(entry.public_support, self.presence_floor) for entry in self.entries],
            dtype=np.float64,
        )
        entry_counts = np.zeros(
            (len(self.entries), self.card_vocab_size + 1),
            dtype=np.int16,
        )
        pseudo_counts = np.zeros(self.card_vocab_size + 1, dtype=np.float64)
        for row, (entry, weight) in enumerate(
            zip(
                self.entries,
                self._support_weights,
                strict=True,
            )
        ):
            card_ids = np.asarray(entry.card_ids, dtype=np.intp)
            counts = np.asarray(entry.counts, dtype=np.int16)
            entry_counts[row, card_ids] = counts
            pseudo_counts[card_ids] += weight * counts
        self._entry_counts = entry_counts
        self._entry_counts_float = entry_counts.astype(np.float64)
        support_total = float(self._support_weights.sum())
        exact_prior_mass = 1.0 - self.unknown_prior_mass
        self._exact_log_priors = np.asarray(
            [
                math.log(exact_prior_mass * float(support) / support_total)
                for support in self._support_weights
            ],
            dtype=np.float64,
        )
        self._log_combinations = np.full(
            (DECK_SIZE + 1, DECK_SIZE + 1),
            -math.inf,
            dtype=np.float64,
        )
        for available in range(DECK_SIZE + 1):
            self._log_combinations[available, : available + 1] = tuple(
                math.log(math.comb(available, observed))
                for observed in range(available + 1)
            )
        self._log_factorials = np.asarray(
            [math.lgamma(count + 1.0) for count in range(DECK_SIZE + 1)],
            dtype=np.float64,
        )
        denominator = float(pseudo_counts[1:].sum()) + float(self.card_vocab_size)
        self._unknown_card_probabilities = (pseudo_counts[1:] + 1.0) / denominator
        self._unknown_log_card_probabilities = np.asarray(
            [
                math.log(float(probability))
                for probability in self._unknown_card_probabilities
            ],
            dtype=np.float64,
        )
        self._unknown_log_prior = math.log(self.unknown_prior_mass)
        for array in (
            self._support_weights,
            self._entry_counts,
            self._entry_counts_float,
            self._exact_log_priors,
            self._log_combinations,
            self._log_factorials,
            self._unknown_card_probabilities,
            self._unknown_log_card_probabilities,
        ):
            array.flags.writeable = False
        self.fingerprint = canonical_belief_fingerprint(
            _CATALOG_FINGERPRINT_DOMAIN,
            self.semantic_payload(),
        )

    def semantic_payload(self) -> dict[str, object]:
        """Return the canonical JSON-safe catalog identity payload."""
        return {
            "schema": PUBLIC_DECK_CATALOG_SCHEMA,
            "unknown_prior_mass": self.unknown_prior_mass,
            "presence_floor": self.presence_floor,
            "card_vocab_size": self.card_vocab_size,
            "card_catalog_fingerprint": self.card_catalog_fingerprint,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "entries": [
                {
                    "signature": entry.signature,
                    "card_ids": list(entry.card_ids),
                    "counts": list(entry.counts),
                    "public_support": entry.public_support,
                }
                for entry in self.entries
            ],
        }

    def native_arrays(self) -> NativePublicDeckCatalogArrays:
        """Export an owned read-only snapshot for one native encoder lifetime.

        The snapshot deliberately does not alias catalog internals. A native
        bridge may retain its arrays for pointer stability without exposing
        mutable catalog state back to Python callers.
        """
        return NativePublicDeckCatalogArrays(
            entry_counts=_immutable_int16_copy(self._entry_counts),
            exact_log_priors=_immutable_float64_copy(self._exact_log_priors),
            log_combinations=_immutable_float64_copy(self._log_combinations),
            log_factorials=_immutable_float64_copy(self._log_factorials),
            unknown_card_probabilities=_immutable_float64_copy(
                self._unknown_card_probabilities
            ),
            unknown_log_card_probabilities=_immutable_float64_copy(
                self._unknown_log_card_probabilities
            ),
            unknown_log_prior=self._unknown_log_prior,
        )

    def posterior(self, known_counts: Counter[int]) -> PublicDeckPosterior:
        """Condition the complete catalog and unknown tail on public evidence."""
        (
            exact_probabilities,
            unknown_probability,
            expected,
            entropy,
            compatible_count,
            known_total,
        ) = self._posterior_components(known_counts)
        return PublicDeckPosterior(
            entries=tuple(
                CatalogPosteriorEntry(
                    signature=entry.signature,
                    probability=probability,
                )
                for entry, probability in zip(
                    self.entries,
                    exact_probabilities,
                    strict=True,
                )
            ),
            unknown_probability=unknown_probability,
            expected_remaining=tuple(
                ExpectedRemainingCard(
                    card_id=card_id,
                    expected_count=float(expected[card_id]),
                )
                for card_id in range(1, self.card_vocab_size + 1)
                if expected[card_id] > 0.0
            ),
            entropy=entropy,
            compatible_deck_count=compatible_count,
            public_evidence_count=known_total,
        )

    def posterior_arrays(
        self,
        known_counts: Counter[int],
    ) -> PublicDeckPosteriorArrays:
        """Condition the catalog directly into compact numeric model columns."""
        (
            _exact_probabilities,
            unknown_probability,
            expected,
            entropy,
            compatible_count,
            known_total,
        ) = self._posterior_components(known_counts)
        card_ids = np.flatnonzero(expected > 0.0)
        return PublicDeckPosteriorArrays(
            card_ids=card_ids.astype(np.int32, copy=False),
            expected_counts=expected[card_ids].astype(np.float32, copy=False),
            unknown_probability=unknown_probability,
            entropy=entropy,
            compatible_deck_count=compatible_count,
            public_evidence_count=known_total,
        )

    def posterior_arrays_many(
        self,
        known_counts: Sequence[Mapping[int, int] | tuple[tuple[int, int], ...]],
    ) -> tuple[PublicDeckPosteriorArrays, ...]:
        """Condition many evidence identities, evaluating each distinct one once."""
        cached: dict[
            tuple[tuple[int, int], ...],
            PublicDeckPosteriorArrays,
        ] = {}
        results: list[PublicDeckPosteriorArrays] = []
        for values in known_counts:
            raw_counts = values if isinstance(values, Mapping) else dict(values)
            known = _validated_known_counts(
                raw_counts,
                card_vocab_size=self.card_vocab_size,
            )
            identity = tuple(sorted(known.items()))
            posterior = cached.get(identity)
            if posterior is None:
                posterior = self.posterior_arrays(Counter(dict(identity)))
                cached[identity] = posterior
            results.append(posterior)
        return tuple(results)

    def _posterior_components(
        self,
        known_counts: Counter[int],
    ) -> tuple[list[float], float, npt.NDArray[np.float64], float, int, int]:
        """Return shared numeric posterior components without row objects."""
        known = _validated_known_counts(
            known_counts,
            card_vocab_size=self.card_vocab_size,
        )
        known_card_ids = np.fromiter(
            known.keys(),
            dtype=np.intp,
            count=len(known),
        )
        known_values = np.fromiter(
            known.values(),
            dtype=np.int64,
            count=len(known),
        )
        known_total = int(known_values.sum(dtype=np.int64))
        entry_known_counts = self._entry_counts[:, known_card_ids]
        compatible = np.all(
            entry_known_counts >= known_values[np.newaxis, :],
            axis=1,
        )
        exact_log_scores = self._exact_log_priors.copy()
        if known_card_ids.size:
            exact_log_scores += self._log_combinations[
                entry_known_counts,
                known_values[np.newaxis, :],
            ].sum(axis=1)
        exact_log_scores -= self._log_combinations[DECK_SIZE, known_total]
        exact_log_scores[~compatible] = -math.inf

        compatible_count = int(np.count_nonzero(compatible))
        if compatible_count == 0:
            exact_probability_array = np.zeros(len(self.entries), dtype=np.float64)
            unknown_probability = 1.0
        else:
            unknown_log_score = self._unknown_log_score(
                known_card_ids,
                known_values,
                known_total=known_total,
            )
            maximum = max(
                float(np.max(exact_log_scores[compatible])),
                unknown_log_score,
            )
            exact_weights = np.zeros(len(self.entries), dtype=np.float64)
            exact_weights[compatible] = np.exp(exact_log_scores[compatible] - maximum)
            unknown_weight = math.exp(unknown_log_score - maximum)
            total = float(exact_weights.sum(dtype=np.float64)) + unknown_weight
            if total <= 0.0 or not math.isfinite(total):
                raise RuntimeError("public catalog posterior normalization failed")
            exact_probability_array = exact_weights / total
            unknown_probability = unknown_weight / total

        expected = exact_probability_array @ self._entry_counts_float
        if known_card_ids.size:
            exact_remaining_known = entry_known_counts - known_values[np.newaxis, :]
            expected[known_card_ids] = exact_probability_array @ exact_remaining_known
        remaining = DECK_SIZE - known_total
        expected[1:] += (
            unknown_probability * float(remaining) * self._unknown_card_probabilities
        )
        exact_probabilities = exact_probability_array.tolist()
        positive_exact = exact_probability_array[exact_probability_array > 0.0]
        entropy = -float(
            np.dot(positive_exact, np.log(positive_exact))
            + (
                unknown_probability * math.log(unknown_probability)
                if unknown_probability > 0.0
                else 0.0
            )
        )
        return (
            exact_probabilities,
            unknown_probability,
            expected,
            entropy,
            compatible_count,
            known_total,
        )

    def _unknown_log_score(
        self,
        known_card_ids: npt.NDArray[np.intp],
        known_values: npt.NDArray[np.int64],
        *,
        known_total: int,
    ) -> float:
        """Return the Laplace-smoothed unknown multinomial log probability."""
        return (
            self._unknown_log_prior
            + float(self._log_factorials[known_total])
            - float(self._log_factorials[known_values].sum(dtype=np.float64))
            + float(
                known_values @ self._unknown_log_card_probabilities[known_card_ids - 1]
            )
        )

    @classmethod
    def from_config(cls, config: PublicDeckCatalogConfig) -> PublicDeckCatalog:
        """Load and verify the exact catalog required by a runtime bundle."""
        catalog, manifest = load_public_deck_catalog(config.manifest_path)
        if manifest.catalog_fingerprint != config.catalog_fingerprint:
            raise ValueError("public catalog config fingerprint mismatch")
        return catalog


def _immutable_int16_copy(
    values: npt.NDArray[np.int16],
) -> npt.NDArray[np.int16]:
    """Copy an int16 array onto immutable bytes while preserving its shape."""
    return np.frombuffer(
        values.tobytes(order="C"),
        dtype=np.int16,
    ).reshape(values.shape)


def _immutable_float64_copy(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Copy a float64 array onto immutable bytes while preserving its shape."""
    return np.frombuffer(
        values.tobytes(order="C"),
        dtype=np.float64,
    ).reshape(values.shape)


def build_public_deck_catalog_from_summary(
    snapshot_path: Path,
    *,
    unknown_prior_mass: float,
    card_catalog_fingerprint: str,
    card_vocab_size: int = DEFAULT_NUM_CARD_IDS,
    presence_floor: int = 1,
) -> PublicDeckCatalog:
    """Build all exact decks in one authoritative public summary snapshot."""
    support_by_signature: Counter[str] = Counter()
    seen_signatures: set[str] = set()
    with snapshot_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            signature = str(row.get("deck_signature", "")).strip()
            if not signature:
                continue
            parse_canonical_signature(signature, max_card_id=card_vocab_size)
            support = _public_support(row)
            support_by_signature[signature] += support
            seen_signatures.add(signature)
    if not seen_signatures:
        raise ValueError("public summary snapshot contains no exact decks")
    entries = tuple(
        PublicDeckCatalogEntry.from_signature(
            signature,
            public_support=int(support_by_signature[signature]),
            card_vocab_size=card_vocab_size,
        )
        for signature in sorted(seen_signatures)
    )
    return PublicDeckCatalog(
        entries,
        unknown_prior_mass=unknown_prior_mass,
        presence_floor=presence_floor,
        card_vocab_size=card_vocab_size,
        card_catalog_fingerprint=card_catalog_fingerprint,
        source_snapshot_sha256=file_sha256(snapshot_path),
    )


def estimate_unknown_prior_mass_from_summary(snapshot_path: Path) -> float:
    """Estimate unseen exact-deck mass with the snapshot's Good-Turing tail.

    The estimator is ``N1 / N`` where ``N1`` is public support belonging to
    exact signatures observed once and ``N`` is total positive public support.
    It is derived solely from the same authoritative catalog snapshot and
    remains fixed throughout a training/deployment bundle.
    """
    support_by_signature: Counter[str] = Counter()
    with snapshot_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            signature = str(row.get("deck_signature", "")).strip()
            if not signature:
                continue
            support_by_signature[signature] += _public_support(row)
    total_support = sum(support_by_signature.values())
    singleton_support = sum(
        support for support in support_by_signature.values() if support == 1
    )
    if total_support <= 0:
        raise ValueError("public summary has no positive support for tail estimation")
    estimate = singleton_support / float(total_support)
    if not 0.0 < estimate < 1.0:
        raise ValueError(
            "Good-Turing unknown tail is degenerate; declare a newer "
            "authoritative snapshot rather than tuning it in RL"
        )
    return estimate


def write_public_deck_catalog(
    catalog: PublicDeckCatalog,
    *,
    artifact_path: Path,
    manifest_path: Path,
) -> PublicDeckCatalogManifest:
    """Atomically publish a compact NPZ and its fingerprint manifest."""
    if artifact_path.suffix != ".npz":
        raise ValueError("public catalog artifact must use the .npz suffix")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(__file__).resolve().parents[3] / "tmp" / "public_catalog"
    temporary_root.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary_artifact = temporary_root / f"{nonce}.npz"
    temporary_manifest = temporary_root / f"{nonce}.json"
    try:
        arrays = _catalog_arrays(catalog.entries)
        with temporary_artifact.open("wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        artifact_sha256 = file_sha256(temporary_artifact)
        manifest = PublicDeckCatalogManifest(
            artifact_filename=artifact_path.name,
            artifact_sha256=artifact_sha256,
            catalog_fingerprint=catalog.fingerprint,
            source_snapshot_sha256=catalog.source_snapshot_sha256,
            card_catalog_fingerprint=catalog.card_catalog_fingerprint,
            card_vocab_size=catalog.card_vocab_size,
            deck_count=len(catalog.entries),
            presence_floor=catalog.presence_floor,
            unknown_prior_mass=catalog.unknown_prior_mass,
        )
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(
                manifest.model_dump(mode="json", by_alias=True),
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_artifact, artifact_path)
        os.replace(temporary_manifest, manifest_path)
        return manifest
    finally:
        temporary_artifact.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def load_public_deck_catalog(
    manifest_path: Path,
) -> tuple[PublicDeckCatalog, PublicDeckCatalogManifest]:
    """Load a catalog only after artifact and semantic fingerprint checks."""
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = PublicDeckCatalogManifest.model_validate(json.load(handle))
    artifact_path = manifest_path.parent / manifest.artifact_filename
    if file_sha256(artifact_path) != manifest.artifact_sha256:
        raise ValueError("public catalog artifact SHA-256 mismatch")
    with np.load(artifact_path, allow_pickle=False) as arrays:
        entries = _entries_from_arrays(arrays)
    catalog = PublicDeckCatalog(
        entries,
        unknown_prior_mass=manifest.unknown_prior_mass,
        presence_floor=manifest.presence_floor,
        card_vocab_size=manifest.card_vocab_size,
        card_catalog_fingerprint=manifest.card_catalog_fingerprint,
        source_snapshot_sha256=manifest.source_snapshot_sha256,
    )
    if len(entries) != manifest.deck_count:
        raise ValueError("public catalog deck count differs from manifest")
    if catalog.fingerprint != manifest.catalog_fingerprint:
        raise ValueError("public catalog semantic fingerprint mismatch")
    return (catalog, manifest)


def _catalog_arrays(
    entries: tuple[PublicDeckCatalogEntry, ...],
) -> dict[str, npt.NDArray[np.generic]]:
    """Flatten sparse entries into compact array storage."""
    offsets = [0]
    card_ids: list[int] = []
    counts: list[int] = []
    for entry in entries:
        card_ids.extend(entry.card_ids)
        counts.extend(entry.counts)
        offsets.append(len(card_ids))
    signature_width = max(len(entry.signature) for entry in entries)
    return {
        "schema_version": np.asarray([1], dtype=np.int16),
        "signatures": np.asarray(
            [entry.signature for entry in entries],
            dtype=f"U{signature_width}",
        ),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "card_ids": np.asarray(card_ids, dtype=np.int32),
        "counts": np.asarray(counts, dtype=np.int16),
        "public_support": np.asarray(
            [entry.public_support for entry in entries],
            dtype=np.int64,
        ),
    }


def _entries_from_arrays(
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> tuple[PublicDeckCatalogEntry, ...]:
    """Reconstruct validated entries from compact array storage."""
    required = {
        "schema_version",
        "signatures",
        "offsets",
        "card_ids",
        "counts",
        "public_support",
    }
    if set(arrays) != required:
        raise ValueError("public catalog artifact arrays differ from schema")
    schema_version = np.asarray(arrays["schema_version"])
    if schema_version.tolist() != [1]:
        raise ValueError("unsupported public catalog array schema version")
    signatures = np.asarray(arrays["signatures"])
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    card_ids = np.asarray(arrays["card_ids"], dtype=np.int64)
    counts = np.asarray(arrays["counts"], dtype=np.int64)
    supports = np.asarray(arrays["public_support"], dtype=np.int64)
    if (
        signatures.ndim != 1
        or offsets.shape != (len(signatures) + 1,)
        or supports.shape != signatures.shape
        or card_ids.shape != counts.shape
        or offsets[0] != 0
        or offsets[-1] != len(card_ids)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("public catalog sparse array shapes are invalid")
    return tuple(
        PublicDeckCatalogEntry(
            signature=str(signatures[row]),
            card_ids=tuple(
                int(value) for value in card_ids[offsets[row] : offsets[row + 1]]
            ),
            counts=tuple(
                int(value) for value in counts[offsets[row] : offsets[row + 1]]
            ),
            public_support=int(supports[row]),
        )
        for row in range(len(signatures))
    )


def _validated_known_counts(
    values: Mapping[int, int],
    *,
    card_vocab_size: int,
) -> Counter[int]:
    """Validate one public evidence multiset in the shared sample space."""
    known: Counter[int] = Counter()
    for raw_card_id, raw_count in values.items():
        card_id = int(raw_card_id)
        count = int(raw_count)
        if card_id <= 0 or card_id > card_vocab_size:
            raise ValueError("public evidence card ID exceeds catalog vocabulary")
        if count <= 0:
            raise ValueError("public evidence counts must be positive")
        known[card_id] = count
    if sum(known.values()) > DECK_SIZE:
        raise ValueError("public evidence cannot exceed a 60-card deck")
    return known


def _public_support(row: Mapping[str, str]) -> int:
    """Read existence support without consulting any win-rate column."""
    for field in ("public_support", "support_count", "games"):
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        value = int(raw)
        if value < 0:
            raise ValueError("public deck support cannot be negative")
        return value
    return 0


def _validate_sha256(value: str, *, label: str) -> None:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} fingerprint must contain 64 hex digits")
