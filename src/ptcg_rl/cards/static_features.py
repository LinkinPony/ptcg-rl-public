"""Build fixed static card feature tables from the bundled game engine.

The engine's card and attack data are the source of truth for intrinsic card
metadata. This module keeps the feature schema explicit so policy/value models
can share the same ``static_feat[card_id]`` table everywhere.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

CARD_TYPE_COUNT = 7
ENERGY_TYPE_COUNT = 12
RETREAT_COST_COUNT = 5
STAGE_COUNT = 3
ATTACK_SLOT_COUNT = 2
ATTACK_SLOT_WIDTH = 16
FEATURE_SIZE = 109
DEFAULT_NUM_CARD_IDS = 1_267
SCHEMA_VERSION = "card_static_features.v1"
StaticFeatureMmapMode = Literal["r", "r+", "c"]

POKEMON_CARD_TYPE = 0
ITEM_CARD_TYPE = 1
TOOL_CARD_TYPE = 2
SUPPORTER_CARD_TYPE = 3
STADIUM_CARD_TYPE = 4
BASIC_ENERGY_CARD_TYPE = 5
SPECIAL_ENERGY_CARD_TYPE = 6

COLORLESS_ENERGY_TYPE = 0

HP_NORMALIZER = 400.0
MAX_RETREAT_COST = 4
MAX_ABILITY_COUNT = 2
MAX_ATTACK_COST = 5
MAX_PRINTED_DAMAGE = 350.0

FEATURE_OFFSETS: Mapping[str, tuple[int, int]] = {
    "card_type": (0, 7),
    "ace_spec": (7, 8),
    "hp_norm": (8, 9),
    "pokemon_type": (9, 21),
    "weakness": (21, 33),
    "has_weakness": (33, 34),
    "resistance": (34, 46),
    "has_resistance": (46, 47),
    "retreat_cost": (47, 52),
    "stage": (52, 55),
    "ex_mega_ex_tera": (55, 58),
    "has_evolves_from": (58, 59),
    "has_ability": (59, 60),
    "ability_count_norm": (60, 61),
    "attack_1_exists": (61, 62),
    "attack_1_cost": (62, 74),
    "attack_1_total_cost_norm": (74, 75),
    "attack_1_printed_damage_norm": (75, 76),
    "attack_1_has_text_effect": (76, 77),
    "attack_2_exists": (77, 78),
    "attack_2_cost": (78, 90),
    "attack_2_total_cost_norm": (90, 91),
    "attack_2_printed_damage_norm": (91, 92),
    "attack_2_has_text_effect": (92, 93),
    "is_basic_energy": (93, 94),
    "is_special_energy": (94, 95),
    "provides": (95, 107),
    "special_has_text": (107, 108),
    "trainer_has_text": (108, 109),
}

_VECTOR_LENGTHS: Mapping[str, int] = {
    "card_type": CARD_TYPE_COUNT,
    "pokemon_type": ENERGY_TYPE_COUNT,
    "weakness": ENERGY_TYPE_COUNT,
    "resistance": ENERGY_TYPE_COUNT,
    "retreat_cost": RETREAT_COST_COUNT,
    "stage": STAGE_COUNT,
    "attack_1_cost": ENERGY_TYPE_COUNT,
    "attack_2_cost": ENERGY_TYPE_COUNT,
    "provides": ENERGY_TYPE_COUNT,
}
_ONE_HOT_FIELDS = {
    "card_type",
    "pokemon_type",
    "weakness",
    "resistance",
    "retreat_cost",
    "stage",
    "provides",
}
_COST_FIELDS = {"attack_1_cost", "attack_2_cost"}
_BINARY_SCALAR_FIELDS = {
    "ace_spec",
    "has_weakness",
    "has_resistance",
    "ex",
    "mega_ex",
    "tera",
    "has_evolves_from",
    "has_ability",
    "attack_1_exists",
    "attack_1_has_text_effect",
    "attack_2_exists",
    "attack_2_has_text_effect",
    "is_basic_energy",
    "is_special_energy",
    "special_has_text",
    "trainer_has_text",
}
_NORMALIZED_SCALAR_FIELDS = {
    "hp_norm",
    "ability_count_norm",
    "attack_1_total_cost_norm",
    "attack_1_printed_damage_norm",
    "attack_2_total_cost_norm",
    "attack_2_printed_damage_norm",
}


class _SkillLike(Protocol):
    text: str


class _CardDataLike(Protocol):
    cardId: int  # noqa: N815
    name: str
    cardType: int  # noqa: N815
    retreatCost: int  # noqa: N815
    hp: int
    weakness: int | None
    resistance: int | None
    energyType: int  # noqa: N815
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    megaEx: bool  # noqa: N815
    tera: bool
    aceSpec: bool  # noqa: N815
    evolvesFrom: str | None  # noqa: N815
    skills: Sequence[_SkillLike]
    attacks: Sequence[int]


class _AttackLike(Protocol):
    attackId: int  # noqa: N815
    text: str
    damage: int
    energies: Sequence[int]


class AttackFeatureSlot(BaseModel):
    """Fixed-width feature block for one printed attack slot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exists: float = 0.0
    cost: tuple[float, ...] = Field(default_factory=lambda: _zeros(ENERGY_TYPE_COUNT))
    total_cost_norm: float = 0.0
    printed_damage_norm: float = 0.0
    has_text_effect: float = 0.0

    @field_validator("exists", "has_text_effect")
    @classmethod
    def valid_binary(cls, value: float) -> float:
        """Reject non-binary attack scalar values."""
        return _validate_binary(value)

    @field_validator("total_cost_norm", "printed_damage_norm")
    @classmethod
    def valid_unit_interval(cls, value: float) -> float:
        """Reject invalid normalized attack scalar values."""
        return _validate_unit_interval(value)

    @field_validator("cost", mode="before")
    @classmethod
    def coerce_cost(cls, value: Sequence[float]) -> tuple[float, ...]:
        """Store cost vectors as tuples for immutable rows."""
        return tuple(float(item) for item in value)

    @field_validator("cost")
    @classmethod
    def valid_cost(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject malformed energy cost vectors."""
        if len(value) != ENERGY_TYPE_COUNT:
            raise ValueError(f"cost must have {ENERGY_TYPE_COUNT} elements")
        for item in value:
            if not np.isfinite(item) or item < 0.0 or item > MAX_ATTACK_COST:
                raise ValueError("cost values must be finite counts in [0, 5]")
        return value

    def to_vector(self) -> tuple[float, ...]:
        """Return this attack slot in schema order."""
        return (
            self.exists,
            *self.cost,
            self.total_cost_norm,
            self.printed_damage_norm,
            self.has_text_effect,
        )

    @classmethod
    def empty(cls) -> AttackFeatureSlot:
        """Return an all-zero attack slot."""
        return cls()

    @classmethod
    def from_attack(cls, attack: _AttackLike) -> AttackFeatureSlot:
        """Build an attack slot from engine attack data."""
        energies = [_to_int(energy) for energy in attack.energies]
        cost = [0.0] * ENERGY_TYPE_COUNT
        for energy in energies:
            _validate_energy_index(energy)
            cost[energy] += 1.0
        return cls(
            exists=1.0,
            cost=tuple(cost),
            total_cost_norm=_normalized(len(energies), MAX_ATTACK_COST),
            printed_damage_norm=_normalized(attack.damage, MAX_PRINTED_DAMAGE),
            has_text_effect=float(bool(attack.text.strip())),
        )


class CardStaticFeatureRow(BaseModel):
    """Validated 109-dimensional static feature row for one card ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    card_type: tuple[float, ...] = Field(default_factory=lambda: _zeros(CARD_TYPE_COUNT))
    ace_spec: float = 0.0
    hp_norm: float = 0.0
    pokemon_type: tuple[float, ...] = Field(
        default_factory=lambda: _zeros(ENERGY_TYPE_COUNT)
    )
    weakness: tuple[float, ...] = Field(
        default_factory=lambda: _zeros(ENERGY_TYPE_COUNT)
    )
    has_weakness: float = 0.0
    resistance: tuple[float, ...] = Field(
        default_factory=lambda: _zeros(ENERGY_TYPE_COUNT)
    )
    has_resistance: float = 0.0
    retreat_cost: tuple[float, ...] = Field(
        default_factory=lambda: _zeros(RETREAT_COST_COUNT)
    )
    stage: tuple[float, ...] = Field(default_factory=lambda: _zeros(STAGE_COUNT))
    ex: float = 0.0
    mega_ex: float = 0.0
    tera: float = 0.0
    has_evolves_from: float = 0.0
    has_ability: float = 0.0
    ability_count_norm: float = 0.0
    attack_1_exists: float = 0.0
    attack_1_cost: tuple[float, ...] = Field(
        default_factory=lambda: _zeros(ENERGY_TYPE_COUNT)
    )
    attack_1_total_cost_norm: float = 0.0
    attack_1_printed_damage_norm: float = 0.0
    attack_1_has_text_effect: float = 0.0
    attack_2_exists: float = 0.0
    attack_2_cost: tuple[float, ...] = Field(
        default_factory=lambda: _zeros(ENERGY_TYPE_COUNT)
    )
    attack_2_total_cost_norm: float = 0.0
    attack_2_printed_damage_norm: float = 0.0
    attack_2_has_text_effect: float = 0.0
    is_basic_energy: float = 0.0
    is_special_energy: float = 0.0
    provides: tuple[float, ...] = Field(
        default_factory=lambda: _zeros(ENERGY_TYPE_COUNT)
    )
    special_has_text: float = 0.0
    trainer_has_text: float = 0.0

    @field_validator(*_VECTOR_LENGTHS.keys(), mode="before")
    @classmethod
    def coerce_vector(cls, value: Sequence[float]) -> tuple[float, ...]:
        """Store vectors as tuples for immutable rows."""
        return tuple(float(item) for item in value)

    @field_validator(*_VECTOR_LENGTHS.keys())
    @classmethod
    def valid_vector(
        cls,
        value: tuple[float, ...],
        info: ValidationInfo,
    ) -> tuple[float, ...]:
        """Reject vectors with bad length, ranges, or one-hot structure."""
        if info.field_name is None:
            raise ValueError("missing field name during vector validation")
        field_name = info.field_name
        expected = _VECTOR_LENGTHS[field_name]
        if len(value) != expected:
            raise ValueError(f"{field_name} must have {expected} elements")
        for item in value:
            if not np.isfinite(item):
                raise ValueError(f"{field_name} contains a non-finite value")
        if field_name in _ONE_HOT_FIELDS:
            _validate_sparse_one_hot(value, field_name)
        if field_name in _COST_FIELDS:
            _validate_cost_counts(value, field_name)
        return value

    @field_validator(*_BINARY_SCALAR_FIELDS)
    @classmethod
    def valid_binary(cls, value: float, info: ValidationInfo) -> float:
        """Reject non-binary scalar values."""
        del info
        return _validate_binary(value)

    @field_validator(*_NORMALIZED_SCALAR_FIELDS)
    @classmethod
    def valid_unit_interval(cls, value: float, info: ValidationInfo) -> float:
        """Reject normalized scalars outside [0, 1]."""
        del info
        return _validate_unit_interval(value)

    def to_vector(self) -> npt.NDArray[np.float32]:
        """Return the dense row as ``float32[F]`` in schema order."""
        values = (
            *self.card_type,
            self.ace_spec,
            self.hp_norm,
            *self.pokemon_type,
            *self.weakness,
            self.has_weakness,
            *self.resistance,
            self.has_resistance,
            *self.retreat_cost,
            *self.stage,
            self.ex,
            self.mega_ex,
            self.tera,
            self.has_evolves_from,
            self.has_ability,
            self.ability_count_norm,
            self.attack_1_exists,
            *self.attack_1_cost,
            self.attack_1_total_cost_norm,
            self.attack_1_printed_damage_norm,
            self.attack_1_has_text_effect,
            self.attack_2_exists,
            *self.attack_2_cost,
            self.attack_2_total_cost_norm,
            self.attack_2_printed_damage_norm,
            self.attack_2_has_text_effect,
            self.is_basic_energy,
            self.is_special_energy,
            *self.provides,
            self.special_has_text,
            self.trainer_has_text,
        )
        if len(values) != FEATURE_SIZE:
            raise ValueError(f"static feature row has {len(values)} values")
        return np.asarray(values, dtype=np.float32)

    @classmethod
    def from_card(
        cls,
        card: _CardDataLike,
        attack_by_id: Mapping[int, _AttackLike],
    ) -> CardStaticFeatureRow:
        """Build a static feature row from one engine ``CardData`` record."""
        card_type = _to_int(card.cardType)
        _validate_card_type(card_type)
        fields: dict[str, Any] = {
            "card_type": _one_hot(card_type, CARD_TYPE_COUNT),
            "ace_spec": float(card.aceSpec),
        }
        if card_type == POKEMON_CARD_TYPE:
            fields.update(_pokemon_fields(card, attack_by_id))
        elif card_type == BASIC_ENERGY_CARD_TYPE:
            fields.update(
                {
                    "is_basic_energy": 1.0,
                    "provides": _one_hot(_to_int(card.energyType), ENERGY_TYPE_COUNT),
                }
            )
        elif card_type == SPECIAL_ENERGY_CARD_TYPE:
            fields.update(
                {
                    "is_special_energy": 1.0,
                    "provides": _one_hot(_to_int(card.energyType), ENERGY_TYPE_COUNT),
                    "special_has_text": float(_has_text(card.skills)),
                }
            )
        else:
            fields["trainer_has_text"] = float(_has_text(card.skills))
        return cls(**fields)


class CardStaticFeatureManifest(BaseModel):
    """Small manifest saved next to a static card feature table."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    created_at_utc: str
    feature_size: int = FEATURE_SIZE
    num_card_ids: int
    row_count: int
    dtype: str = "float32"
    pad_oov_row: int = 0
    feature_offsets: dict[str, tuple[int, int]]
    normalization: dict[str, float]
    source: dict[str, Any]


class StaticFeatureBuildConfig(BaseModel):
    """Hydra-backed config for static card feature table builds."""

    model_config = ConfigDict(extra="forbid")

    output_path: Path = Path("outputs/cards/static_features/card_static_features.npz")
    manifest_path: Path | None = None
    parquet_path: Path | None = None
    npy_path: Path | None = None
    num_card_ids: int = DEFAULT_NUM_CARD_IDS

    @field_validator("num_card_ids")
    @classmethod
    def valid_num_card_ids(cls, value: int) -> int:
        """Reject non-positive card-ID table sizes."""
        if value <= 0:
            raise ValueError("num_card_ids must be positive")
        return value


def build_static_feature_table(
    cards: Sequence[_CardDataLike],
    attacks: Sequence[_AttackLike],
    *,
    num_card_ids: int | None = DEFAULT_NUM_CARD_IDS,
) -> npt.NDArray[np.float32]:
    """Build ``float32[num_card_ids + 1, 109]`` static features.

    Row 0 is reserved for pad/OOV and remains all zeros.
    """
    if num_card_ids is None:
        num_card_ids = max(card.cardId for card in cards)
    if num_card_ids <= 0:
        raise ValueError("num_card_ids must be positive")
    table = np.zeros((num_card_ids + 1, FEATURE_SIZE), dtype=np.float32)
    attack_by_id = {attack.attackId: attack for attack in attacks}

    for card in cards:
        if card.cardId <= 0 or card.cardId > num_card_ids:
            raise ValueError(
                f"cardId {card.cardId} is outside configured range 1..{num_card_ids}"
            )
        table[card.cardId] = CardStaticFeatureRow.from_card(
            card,
            attack_by_id,
        ).to_vector()

    validate_static_feature_table(table, num_card_ids=num_card_ids)
    return table


def build_static_features_from_engine(
    *,
    num_card_ids: int | None = DEFAULT_NUM_CARD_IDS,
) -> npt.NDArray[np.float32]:
    """Load bundled engine data and build the static feature table."""
    cards, attacks = load_engine_card_data()
    return build_static_feature_table(cards, attacks, num_card_ids=num_card_ids)


def load_engine_card_data() -> tuple[Sequence[_CardDataLike], Sequence[_AttackLike]]:
    """Return ``(all_card_data(), all_attack())`` from the bundled engine."""
    try:
        from ptcg_rl.engine.runtime import load_cg_api

        cg_api = load_cg_api()
    except ImportError as exc:
        raise ImportError(
            "Could not import cg.api. Run with "
            "PYTHONPATH=data/sample_submission:src or from a Kaggle package."
        ) from exc
    typed_cg_api = cast(Any, cg_api)
    all_card_data = typed_cg_api.all_card_data
    all_attack = typed_cg_api.all_attack
    cards = cast(Sequence[_CardDataLike], all_card_data())
    attacks = cast(Sequence[_AttackLike], all_attack())
    return cards, attacks


def validate_static_feature_table(
    static_feat: npt.NDArray[np.float32],
    *,
    num_card_ids: int | None = None,
) -> None:
    """Validate shape, dtype, finite values, and all-zero pad/OOV row."""
    if static_feat.ndim != 2:
        raise ValueError("static_feat must be a rank-2 array")
    if static_feat.shape[1] != FEATURE_SIZE:
        raise ValueError(f"static_feat must have {FEATURE_SIZE} feature columns")
    if num_card_ids is not None and static_feat.shape[0] != num_card_ids + 1:
        raise ValueError(
            f"static_feat must have {num_card_ids + 1} rows for num_card_ids"
        )
    if static_feat.dtype != np.float32:
        raise ValueError("static_feat dtype must be float32")
    if not np.isfinite(static_feat).all():
        raise ValueError("static_feat contains NaN or infinite values")
    if not np.all(static_feat[0] == 0.0):
        raise ValueError("static_feat row 0 must be all zeros for pad/OOV")


def safe_card_id_indices(
    card_ids: Sequence[int] | npt.NDArray[np.integer[Any]],
    *,
    num_card_ids: int,
) -> npt.NDArray[np.int64]:
    """Map invalid or unknown card IDs to the pad/OOV row index 0."""
    values = np.asarray(card_ids, dtype=np.int64)
    return np.where((values >= 1) & (values <= num_card_ids), values, 0).astype(
        np.int64,
        copy=False,
    )


def features_for_card_ids(
    static_feat: npt.NDArray[np.float32],
    card_ids: Sequence[int] | npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.float32]:
    """Return static feature rows, mapping OOV IDs to row 0."""
    validate_static_feature_table(static_feat)
    indices = safe_card_id_indices(card_ids, num_card_ids=static_feat.shape[0] - 1)
    return static_feat[indices]


def write_static_feature_artifacts(
    static_feat: npt.NDArray[np.float32],
    output_path: Path,
    *,
    manifest: CardStaticFeatureManifest,
    manifest_path: Path | None = None,
    parquet_path: Path | None = None,
    npy_path: Path | None = None,
) -> None:
    """Write compact feature-table artifacts plus JSON manifest."""
    validate_static_feature_table(
        static_feat,
        num_card_ids=manifest.num_card_ids,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        static_feat=static_feat,
        card_ids=np.arange(static_feat.shape[0], dtype=np.int32),
        schema_version=np.asarray(SCHEMA_VERSION),
    )
    resolved_manifest_path = manifest_path or default_manifest_path(output_path)
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if parquet_path is not None:
        write_static_feature_parquet(static_feat, parquet_path)
    if npy_path is not None:
        write_static_feature_npy(static_feat, npy_path)


def write_static_feature_npy(
    static_feat: npt.NDArray[np.float32],
    npy_path: Path,
) -> None:
    """Write an optional NPY table that can be memory-mapped at act time."""
    validate_static_feature_table(static_feat)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, static_feat)


def write_static_feature_parquet(
    static_feat: npt.NDArray[np.float32],
    parquet_path: Path,
) -> None:
    """Write an optional Parquet copy with one row per card ID."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_static_feature_table(static_feat)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, npt.NDArray[np.float32] | npt.NDArray[np.int32]] = {
        "card_id": np.arange(static_feat.shape[0], dtype=np.int32)
    }
    for feature_index in range(static_feat.shape[1]):
        columns[f"f{feature_index:03d}"] = static_feat[:, feature_index]
    pq.write_table(pa.table(columns), parquet_path)


def load_static_feature_table(
    path: Path,
    *,
    mmap_mode: StaticFeatureMmapMode | None = None,
) -> npt.NDArray[np.float32]:
    """Load and validate a static feature table from NPZ or NPY."""
    if path.suffix.lower() == ".npy":
        static_feat = cast(
            npt.NDArray[np.float32],
            np.load(path, mmap_mode=mmap_mode),
        )
    else:
        with np.load(path) as data:
            static_feat = np.asarray(data["static_feat"], dtype=np.float32)
    validate_static_feature_table(static_feat)
    return static_feat


def default_manifest_path(output_path: Path) -> Path:
    """Return the default manifest path for an NPZ feature table."""
    return output_path.with_suffix(".manifest.json")


def build_manifest(
    static_feat: npt.NDArray[np.float32],
    *,
    source: Mapping[str, Any],
) -> CardStaticFeatureManifest:
    """Create a manifest for a freshly built static feature table."""
    validate_static_feature_table(static_feat)
    return CardStaticFeatureManifest(
        created_at_utc=datetime.now(UTC).isoformat(),
        num_card_ids=static_feat.shape[0] - 1,
        row_count=static_feat.shape[0],
        feature_offsets=dict(FEATURE_OFFSETS),
        normalization={
            "hp": HP_NORMALIZER,
            "ability_count": float(MAX_ABILITY_COUNT),
            "attack_total_cost": float(MAX_ATTACK_COST),
            "printed_damage": MAX_PRINTED_DAMAGE,
        },
        source=dict(source),
    )


def build_and_write_from_engine(config: StaticFeatureBuildConfig) -> dict[str, Any]:
    """Build static card features from engine data and write configured outputs."""
    cards, attacks = load_engine_card_data()
    static_feat = build_static_feature_table(
        cards,
        attacks,
        num_card_ids=config.num_card_ids,
    )
    manifest = build_manifest(
        static_feat,
        source={
            "engine": "cg.api.all_card_data/all_attack",
            "card_count": len(cards),
            "attack_count": len(attacks),
            "max_card_id": max(card.cardId for card in cards),
            "max_attack_id": max(attack.attackId for attack in attacks),
        },
    )
    write_static_feature_artifacts(
        static_feat,
        config.output_path,
        manifest=manifest,
        manifest_path=config.manifest_path,
        parquet_path=config.parquet_path,
        npy_path=config.npy_path,
    )
    report = {
        "output_path": str(config.output_path),
        "manifest_path": str(config.manifest_path or default_manifest_path(config.output_path)),
        "parquet_path": str(config.parquet_path) if config.parquet_path else None,
        "npy_path": str(config.npy_path) if config.npy_path else None,
        "shape": list(static_feat.shape),
        "dtype": str(static_feat.dtype),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _pokemon_fields(
    card: _CardDataLike,
    attack_by_id: Mapping[int, _AttackLike],
) -> dict[str, Any]:
    if len(card.attacks) > ATTACK_SLOT_COUNT:
        raise ValueError(
            f"cardId {card.cardId} has {len(card.attacks)} attacks; "
            f"schema supports {ATTACK_SLOT_COUNT}"
        )
    attack_slots = [_attack_slot(attack_id, attack_by_id) for attack_id in card.attacks]
    while len(attack_slots) < ATTACK_SLOT_COUNT:
        attack_slots.append(AttackFeatureSlot.empty())

    ability_count = len(card.skills)
    return {
        "hp_norm": _normalized(card.hp, HP_NORMALIZER),
        "pokemon_type": _one_hot(_to_int(card.energyType), ENERGY_TYPE_COUNT),
        "weakness": _optional_one_hot(card.weakness, ENERGY_TYPE_COUNT),
        "has_weakness": float(card.weakness is not None),
        "resistance": _optional_one_hot(card.resistance, ENERGY_TYPE_COUNT),
        "has_resistance": float(card.resistance is not None),
        "retreat_cost": _one_hot(
            min(max(card.retreatCost, 0), MAX_RETREAT_COST),
            RETREAT_COST_COUNT,
        ),
        "stage": _stage_one_hot(card),
        "ex": float(card.ex or card.megaEx),
        "mega_ex": float(card.megaEx),
        "tera": float(card.tera),
        "has_evolves_from": float(card.evolvesFrom is not None),
        "has_ability": float(ability_count > 0),
        "ability_count_norm": min(ability_count, MAX_ABILITY_COUNT) / MAX_ABILITY_COUNT,
        "attack_1_exists": attack_slots[0].exists,
        "attack_1_cost": attack_slots[0].cost,
        "attack_1_total_cost_norm": attack_slots[0].total_cost_norm,
        "attack_1_printed_damage_norm": attack_slots[0].printed_damage_norm,
        "attack_1_has_text_effect": attack_slots[0].has_text_effect,
        "attack_2_exists": attack_slots[1].exists,
        "attack_2_cost": attack_slots[1].cost,
        "attack_2_total_cost_norm": attack_slots[1].total_cost_norm,
        "attack_2_printed_damage_norm": attack_slots[1].printed_damage_norm,
        "attack_2_has_text_effect": attack_slots[1].has_text_effect,
    }


def _attack_slot(
    attack_id: int,
    attack_by_id: Mapping[int, _AttackLike],
) -> AttackFeatureSlot:
    try:
        return AttackFeatureSlot.from_attack(attack_by_id[attack_id])
    except KeyError as exc:
        raise ValueError(f"attackId {attack_id} is referenced but missing") from exc


def _stage_one_hot(card: _CardDataLike) -> tuple[float, ...]:
    if card.basic:
        return _one_hot(0, STAGE_COUNT)
    if card.stage1:
        return _one_hot(1, STAGE_COUNT)
    if card.stage2:
        return _one_hot(2, STAGE_COUNT)
    return _zeros(STAGE_COUNT)


def _has_text(skills: Sequence[_SkillLike]) -> bool:
    return any(skill.text.strip() for skill in skills)


def _zeros(length: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(length))


def _one_hot(index: int, length: int) -> tuple[float, ...]:
    if index < 0 or index >= length:
        raise ValueError(f"index {index} outside one-hot length {length}")
    values = [0.0] * length
    values[index] = 1.0
    return tuple(values)


def _optional_one_hot(index: int | None, length: int) -> tuple[float, ...]:
    if index is None:
        return _zeros(length)
    return _one_hot(_to_int(index), length)


def _normalized(value: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("normalization denominator must be positive")
    return min(max(float(value), 0.0), denominator) / denominator


def _to_int(value: int) -> int:
    return int(value)


def _validate_card_type(card_type: int) -> None:
    if card_type < 0 or card_type >= CARD_TYPE_COUNT:
        raise ValueError(f"unknown cardType {card_type}")


def _validate_energy_index(energy: int) -> None:
    if energy < 0 or energy >= ENERGY_TYPE_COUNT:
        raise ValueError(f"unknown EnergyType {energy}")


def _validate_sparse_one_hot(values: Sequence[float], field_name: str) -> None:
    total = 0.0
    for value in values:
        _validate_binary(value)
        total += value
    if total > 1.0:
        raise ValueError(f"{field_name} must be one-hot or all-zero")


def _validate_cost_counts(values: Sequence[float], field_name: str) -> None:
    for value in values:
        if not np.isfinite(value) or value < 0.0 or value > MAX_ATTACK_COST:
            raise ValueError(f"{field_name} contains invalid energy counts")


def _validate_binary(value: float) -> float:
    if not np.isfinite(value) or value not in {0.0, 1.0}:
        raise ValueError("value must be binary 0/1")
    return float(value)


def _validate_unit_interval(value: float) -> float:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("value must be in [0, 1]")
    return float(value)


def main() -> None:
    """Hydra entry point for building static card feature artifacts.

    Hydra is imported lazily so that act-time and Kaggle-side consumers can
    import this module without training-only dependencies installed.
    """
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(
        version_base=None,
        config_path="../../../configs",
        config_name="cards/static_features",
    )
    def _run(hydra_config: DictConfig) -> None:
        raw_config = OmegaConf.to_container(hydra_config, resolve=True)
        if not isinstance(raw_config, dict):
            raise ValueError("Hydra config must resolve to a dictionary.")
        config = StaticFeatureBuildConfig.model_validate(
            cast(dict[str, Any], raw_config)
        )
        build_and_write_from_engine(config)

    _run()


if __name__ == "__main__":
    main()
