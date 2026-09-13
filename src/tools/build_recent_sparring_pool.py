"""Build the July 12 recent-ladder RL deck pool from public replays.

The tool extracts the opponent's exact 60-card list from the public episodes
of the three most recent local submissions. It emits three separate assets:

* a compact Hydra deck-pool config for candidate sparring and opponents;
* a raw, frequency-weighted belief prior with one row per exact signature;
* an evidence manifest recording source counts and each sampling component.

The belief prior intentionally uses observed replay counts only. Counter
emphasis is applied only to the training opponent distribution.

Run from the repository root with::

    PYTHONPATH=data/sample_submission:src \
      python src/tools/build_recent_sparring_pool.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ptcg_rl.data.kaggle_deck import records

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = Path("docs/experiments/rl_71eb_recent_sparring_20260712")
_POOL_CONFIG_PATH = Path("configs/rl/deck_pools/71eb_recent_sparring_20260712.yaml")
_DYNAMIC_POOL_DIR = Path("docs/experiments/rl_dynamic_deck_pool_20260708")
_CARD_DATA_PATH = Path("data/EN_Card_Data.csv")
_OUR_TEAM_NAME = "Marshall Maximizer"
_SUBMISSION_IDS = ("54529356", "54533777", "54599318")
_REPLAY_GLOB = "episode-*-replay.json"

_TARGET_HASH = "71eb9591262a"
_RECENT_OPPONENT_MASS = 0.70
_COUNTER_OPPONENT_MASS = 0.20
_BROAD_OPPONENT_MASS = 0.10

# These are exact, recently observed lists. The weights are normalized within
# the 20% candidate sparring lane; the target lane has its own fixed 80% mass.
_SPARRING_WEIGHTS = {
    "ea6a498ffa65": 0.22,  # Cinderace / Archaludon
    "83a2c0038214": 0.15,  # Mega Abomasnow / Kyogre
    "404e13fcc933": 0.10,  # pure Crustle
    "2d693f385546": 0.08,  # current Mega Lucario tuning
    "1577c503833f": 0.07,  # Iono's Bellibolt
    "4c05a3fbaac9": 0.07,  # Mega Froslass / Mega Starmie
    "b72e3317b0a0": 0.05,  # N's Zoroark
    "86ad798c0833": 0.05,  # Hydrapple / Meganium
    "e894cc25ee12": 0.05,  # Mega Charizard / Blaziken
    "d8bbec675a79": 0.04,  # Mega Mawile / Zacian
    "4714e0ee9a5a": 0.04,  # Mega Diancie / Mega Lopunny
    "fbb974b319ab": 0.03,  # Ethan's Typhlosion / Dragapult
    "797b78a5cef9": 0.03,  # Walrein / Black Kyurem
    "80a3361415b2": 0.02,  # Drednaw / Crustle
}

# The counter overlay is separate from the recent-frequency component. Within
# each bucket, its mass follows the observed exact-signature frequencies.
_COUNTER_BUCKET_MASSES = {
    "cinderace_archaludon": 0.12,
    "cinderace_starmie": 0.04,
    "hops_trevenant": 0.02,
    "iono_bellibolt": 0.02,
}


@dataclass(frozen=True)
class ReplayDeck:
    """One exact opponent deck extracted from a public replay."""

    submission_id: str
    episode_id: int
    signature: str
    cards: tuple[int, ...]


@dataclass(frozen=True)
class DeckAsset:
    """One signature-deduplicated deck available to the generated pool."""

    signature: str
    deck_hash: str
    cards: tuple[int, ...]
    path: Path
    archetype: str
    dynamic_label: str
    live_counts: Counter[str]

    @property
    def observed_games(self) -> int:
        """Return the number of recent public replays using this exact list."""
        return sum(self.live_counts.values())

    @property
    def is_dynamic(self) -> bool:
        """Return whether this exact signature was in the previous pool."""
        return bool(self.dynamic_label)


def _repo_path(path: Path) -> Path:
    return _REPOSITORY_ROOT / path


def _read_replay_decks() -> tuple[ReplayDeck, ...]:
    decks: list[ReplayDeck] = []
    for submission_id in _SUBMISSION_IDS:
        replay_dir = _repo_path(Path("dist/kaggle_logs") / submission_id / "replays")
        replay_paths = sorted(replay_dir.glob(_REPLAY_GLOB))
        if not replay_paths:
            raise FileNotFoundError(f"no public replays found in {replay_dir}")
        for replay_path in replay_paths:
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
            teams = payload.get("info", {}).get("TeamNames", [])
            if teams.count(_OUR_TEAM_NAME) != 1:
                raise ValueError(
                    f"expected one {_OUR_TEAM_NAME!r} seat in {replay_path}: {teams}"
                )
            our_seat = teams.index(_OUR_TEAM_NAME)
            deck_lists = _initial_deck_lists(payload, replay_path)
            cards = tuple(sorted(int(card_id) for card_id in deck_lists[1 - our_seat]))
            _validate_cards(cards, source=replay_path)
            decks.append(
                ReplayDeck(
                    submission_id=submission_id,
                    episode_id=int(payload["info"]["EpisodeId"]),
                    signature=records.deck_signature(list(cards)),
                    cards=cards,
                )
            )
    return tuple(decks)


def _initial_deck_lists(payload: dict[str, Any], replay_path: Path) -> list[Any]:
    steps = payload.get("steps", [])
    if not steps or not steps[0]:
        raise ValueError(f"replay has no initial step: {replay_path}")
    for seat in steps[0]:
        visualizations = seat.get("visualize", [])
        for visualization in visualizations:
            action = visualization.get("action")
            if (
                isinstance(action, list)
                and len(action) == 2
                and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
            ):
                return action
    raise ValueError(f"replay has no exact initial deck lists: {replay_path}")


def _validate_cards(cards: tuple[int, ...], *, source: Path) -> None:
    if len(cards) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(cards)}: {source}")
    if any(card_id <= 0 for card_id in cards):
        raise ValueError(f"deck contains a non-positive card id: {source}")


def _load_dynamic_pool() -> dict[str, tuple[str, tuple[int, ...], Path]]:
    manifest_path = _repo_path(_DYNAMIC_POOL_DIR / "manifest.csv")
    output: dict[str, tuple[str, tuple[int, ...], Path]] = {}
    with manifest_path.open(encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            signature = row["deck_signature"]
            relative_path = _DYNAMIC_POOL_DIR / row["deck_path"]
            cards = tuple(records.read_deck(_repo_path(relative_path)))
            if records.deck_signature(list(cards)) != signature:
                raise ValueError(
                    f"dynamic manifest signature mismatch: {relative_path}"
                )
            output[signature] = (row["deck_label"], cards, relative_path)
    if len(output) != 37:
        raise ValueError(f"expected 37 dynamic signatures, got {len(output)}")
    return output


def _build_assets(
    replay_decks: tuple[ReplayDeck, ...],
    dynamic_pool: dict[str, tuple[str, tuple[int, ...], Path]],
) -> tuple[DeckAsset, ...]:
    live_counts: dict[str, Counter[str]] = {}
    live_cards: dict[str, tuple[int, ...]] = {}
    for deck in replay_decks:
        live_counts.setdefault(deck.signature, Counter())[deck.submission_id] += 1
        previous_cards = live_cards.setdefault(deck.signature, deck.cards)
        if previous_cards != deck.cards:
            raise ValueError(f"signature collision for {deck.signature}")

    card_meta = records.load_card_meta(_repo_path(_CARD_DATA_PATH))
    output: list[DeckAsset] = []
    for signature in sorted(set(live_counts) | set(dynamic_pool)):
        deck_hash = records.signature_hash(signature)
        dynamic_label, dynamic_cards, dynamic_path = dynamic_pool.get(
            signature, ("", (), Path())
        )
        cards = live_cards.get(signature, dynamic_cards)
        _validate_cards(cards, source=dynamic_path or Path(f"replay:{deck_hash}"))
        if dynamic_label:
            path = dynamic_path
        else:
            path = _OUTPUT_DIR / "decks" / f"{deck_hash}.csv"
        output.append(
            DeckAsset(
                signature=signature,
                deck_hash=deck_hash,
                cards=cards,
                path=path,
                archetype=_archetype_label(signature, card_meta),
                dynamic_label=dynamic_label,
                live_counts=live_counts.get(signature, Counter()),
            )
        )
    return tuple(output)


def _archetype_label(
    signature: str,
    card_meta: dict[int, records.CardMeta],
) -> str:
    pokemon: list[tuple[int, int, str]] = []
    for card_id, count in records.signature_counts(signature).items():
        meta = card_meta.get(card_id, records.missing_card(card_id))
        if meta.stage_or_type.endswith("Pokémon"):
            pokemon.append((-count, card_id, meta.name))
    names: list[str] = []
    for _, _, name in sorted(pokemon):
        if name not in names:
            names.append(name)
    return " / ".join(names[:7]) or "unknown"


def _counter_bucket(signature: str) -> str:
    counts = records.signature_counts(signature)
    card_ids = set(counts)
    if {169, 190, 666} <= card_ids:
        return "cinderace_archaludon"
    if {666, 1031} <= card_ids:
        return "cinderace_starmie"
    if 879 in card_ids:
        return "hops_trevenant"
    if 270 in card_ids:
        return "iono_bellibolt"
    return ""


def _opponent_components(
    assets: tuple[DeckAsset, ...],
) -> dict[str, tuple[float, float, float]]:
    recent_total = sum(asset.observed_games for asset in assets)
    dynamic_total = sum(asset.is_dynamic for asset in assets)
    if recent_total <= 0 or dynamic_total != 37:
        raise ValueError("opponent pool has invalid recent or dynamic support")

    bucket_totals: Counter[str] = Counter()
    for asset in assets:
        bucket_totals[_counter_bucket(asset.signature)] += asset.observed_games
    for bucket in _COUNTER_BUCKET_MASSES:
        if bucket_totals[bucket] <= 0:
            raise ValueError(f"counter bucket has no recent decks: {bucket}")

    output: dict[str, tuple[float, float, float]] = {}
    for asset in assets:
        recent = _RECENT_OPPONENT_MASS * asset.observed_games / recent_total
        bucket = _counter_bucket(asset.signature)
        counter = 0.0
        if bucket:
            counter = (
                _COUNTER_BUCKET_MASSES[bucket]
                * asset.observed_games
                / bucket_totals[bucket]
            )
        broad = _BROAD_OPPONENT_MASS / dynamic_total if asset.is_dynamic else 0.0
        output[asset.signature] = (recent, counter, broad)

    total = sum(sum(components) for components in output.values())
    if not math.isclose(total, 1.0, abs_tol=1.0e-12):
        raise ValueError(f"opponent weights sum to {total}, expected 1.0")
    return output


def _write_generated_decks(assets: tuple[DeckAsset, ...]) -> None:
    for asset in assets:
        if asset.is_dynamic:
            continue
        output_path = _repo_path(asset.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(f"{card_id}\n" for card_id in sorted(asset.cards)),
            encoding="utf-8",
        )


def _write_belief_prior(assets: tuple[DeckAsset, ...]) -> Path:
    output_path = _repo_path(_OUTPUT_DIR / "belief_deck_signature_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (asset for asset in assets if asset.observed_games > 0),
        key=lambda asset: (-asset.observed_games, asset.deck_hash),
    )
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=("deck_label", "deck_signature", "games", "win_rate"),
            lineterminator="\n",
        )
        writer.writeheader()
        for asset in rows:
            writer.writerow(
                {
                    "deck_label": f"recent_{asset.deck_hash}",
                    "deck_signature": asset.signature,
                    "games": asset.observed_games,
                    "win_rate": 0.5,
                }
            )
    return output_path


def _write_manifest(
    assets: tuple[DeckAsset, ...],
    opponent_components: dict[str, tuple[float, float, float]],
) -> Path:
    output_path = _repo_path(_OUTPUT_DIR / "manifest.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "deck_hash",
        "archetype",
        "deck_path",
        "source_kind",
        "observed_games",
        "submission_54529356_games",
        "submission_54533777_games",
        "submission_54599318_games",
        "dynamic37",
        "candidate_lane",
        "candidate_sparring_weight",
        "counter_bucket",
        "recent_opponent_weight",
        "counter_opponent_weight",
        "broad_opponent_weight",
        "opponent_weight",
        "deck_signature",
    )
    rows = sorted(
        assets,
        key=lambda asset: (-asset.observed_games, asset.deck_hash),
    )
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for asset in rows:
            recent, counter, broad = opponent_components[asset.signature]
            if asset.deck_hash == _TARGET_HASH:
                candidate_lane = "target"
                candidate_weight = 1.0
            elif asset.deck_hash in _SPARRING_WEIGHTS:
                candidate_lane = "near"
                candidate_weight = _SPARRING_WEIGHTS[asset.deck_hash]
            else:
                candidate_lane = ""
                candidate_weight = 0.0
            writer.writerow(
                {
                    "deck_hash": asset.deck_hash,
                    "archetype": asset.archetype,
                    "deck_path": asset.path.as_posix(),
                    "source_kind": _source_kind(asset),
                    "observed_games": asset.observed_games,
                    "submission_54529356_games": asset.live_counts["54529356"],
                    "submission_54533777_games": asset.live_counts["54533777"],
                    "submission_54599318_games": asset.live_counts["54599318"],
                    "dynamic37": str(asset.is_dynamic).lower(),
                    "candidate_lane": candidate_lane,
                    "candidate_sparring_weight": f"{candidate_weight:.12f}",
                    "counter_bucket": _counter_bucket(asset.signature),
                    "recent_opponent_weight": f"{recent:.12f}",
                    "counter_opponent_weight": f"{counter:.12f}",
                    "broad_opponent_weight": f"{broad:.12f}",
                    "opponent_weight": f"{recent + counter + broad:.12f}",
                    "deck_signature": asset.signature,
                }
            )
    return output_path


def _source_kind(asset: DeckAsset) -> str:
    if asset.observed_games > 0 and asset.is_dynamic:
        return "recent_public_replay+dynamic37"
    if asset.observed_games > 0:
        return "recent_public_replay"
    return "dynamic37_broad_tail"


def _write_pool_config(
    assets: tuple[DeckAsset, ...],
    opponent_components: dict[str, tuple[float, float, float]],
) -> Path:
    by_hash = {asset.deck_hash: asset for asset in assets}
    missing_hashes = ({_TARGET_HASH} | set(_SPARRING_WEIGHTS)) - set(by_hash)
    if missing_hashes:
        raise ValueError(f"candidate hashes are absent from the pool: {missing_hashes}")

    candidate_entries = [
        {
            "path": by_hash[_TARGET_HASH].path.as_posix(),
            "weight": 1.0,
            "label": f"target_{_TARGET_HASH}",
        }
    ]
    candidate_entries.extend(
        {
            "path": by_hash[deck_hash].path.as_posix(),
            "weight": weight,
            "label": f"sparring_{deck_hash}",
        }
        for deck_hash, weight in _SPARRING_WEIGHTS.items()
    )

    opponent_entries: list[dict[str, object]] = []
    for asset in sorted(assets, key=lambda item: item.deck_hash):
        components = opponent_components[asset.signature]
        weight = sum(components)
        if weight <= 0.0:
            continue
        opponent_entries.append(
            {
                "path": asset.path.as_posix(),
                "weight": weight,
                "label": f"opponent_{asset.deck_hash}",
            }
        )

    output_path = _repo_path(_POOL_CONFIG_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "curriculum": {
            "candidate_deck_pool": candidate_entries,
            "opponent_deck_pool": opponent_entries,
        }
    }
    output_path.write_text(
        "# @package _global_\n"
        + yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return output_path


def _write_readme(
    replay_decks: tuple[ReplayDeck, ...],
    assets: tuple[DeckAsset, ...],
) -> Path:
    output_path = _repo_path(_OUTPUT_DIR / "README.md")
    replay_counts = Counter(deck.submission_id for deck in replay_decks)
    observed_assets = tuple(asset for asset in assets if asset.observed_games > 0)
    dynamic_games = sum(
        asset.observed_games for asset in observed_assets if asset.is_dynamic
    )
    near_lines = []
    by_hash = {asset.deck_hash: asset for asset in assets}
    for deck_hash, weight in _SPARRING_WEIGHTS.items():
        asset = by_hash[deck_hash]
        near_lines.append(
            f"| `{deck_hash}` | {asset.archetype} | {asset.observed_games} | "
            f"{weight:.0%} |"
        )
    text = f"""# RL 71eb Recent Sparring Pool 20260712

This pool is built from {len(replay_decks)} public Kaggle episodes downloaded
through the Kaggle CLI for submissions `54529356`, `54533777`, and `54599318`.
The per-submission counts are {replay_counts["54529356"]},
{replay_counts["54533777"]}, and {replay_counts["54599318"]} respectively.
They contain {len(observed_assets)} unique exact opponent signatures; only
{dynamic_games}/{len(replay_decks)} observed games use an exact signature from
the previous 37-deck pool.

## Candidate allocation

The training profile assigns 80% of candidate games to target deck `71eb` and
20% to the near/sparring lane. Weights below are normalized inside that 20%
lane; all are exact public-ladder lists newly observed beyond dynamic37.

| Hash | Archetype | Observed games | Near-lane share |
|---|---|---:|---:|
{chr(10).join(near_lines)}

## Opponent allocation

The generated opponent weights sum to one and are the direct sum of:

- 70% exact recent replay frequency across all observed signatures;
- 20% fixed counter overlay: 12% Cinderace/Archaludon, 4%
  Cinderace/Starmie, 2% Hop's Trevenant, and 2% Iono's Bellibolt;
- 10% uniform broad tail across the previous 37 exact decks.

No win-rate measurement changes these weights during training. The raw
components are recorded in `manifest.csv`.

## Belief prior

`belief_deck_signature_summary.csv` contains one row per exact recently
observed signature. `games` is the integer episode count summed across the
three submissions and therefore sums to {len(replay_decks)}. It contains no
counter overlay and does not use training opponent weights.

All referenced decks expand to exactly 60 cards. Generated deck files are
under `decks/`; existing dynamic37 signatures reuse their immutable source
files. Rebuild with:

```bash
PYTHONPATH=data/sample_submission:src \\
  python src/tools/build_recent_sparring_pool.py
```
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _validate_outputs(
    assets: tuple[DeckAsset, ...],
    belief_path: Path,
    opponent_components: dict[str, tuple[float, float, float]],
) -> None:
    for asset in assets:
        cards = records.read_deck(_repo_path(asset.path))
        if records.deck_signature(cards) != asset.signature:
            raise ValueError(f"written deck signature mismatch: {asset.path}")

    with belief_path.open(encoding="utf-8", newline="") as file_obj:
        belief_rows = list(csv.DictReader(file_obj))
    signatures = [row["deck_signature"] for row in belief_rows]
    if len(signatures) != len(set(signatures)):
        raise ValueError("belief prior contains duplicate exact signatures")
    expected_games = sum(asset.observed_games for asset in assets)
    belief_games = sum(int(row["games"]) for row in belief_rows)
    if belief_games != expected_games:
        raise ValueError(
            f"belief prior games sum to {belief_games}, expected {expected_games}"
        )
    opponent_total = sum(sum(parts) for parts in opponent_components.values())
    if not math.isclose(opponent_total, 1.0, abs_tol=1.0e-12):
        raise ValueError(f"opponent weight total is {opponent_total}")
    if not math.isclose(sum(_SPARRING_WEIGHTS.values()), 1.0, abs_tol=1.0e-12):
        raise ValueError("candidate sparring weights must sum to one")


def main() -> int:
    """Build and validate the recent-ladder training assets."""
    replay_decks = _read_replay_decks()
    dynamic_pool = _load_dynamic_pool()
    assets = _build_assets(replay_decks, dynamic_pool)
    opponent_components = _opponent_components(assets)
    _write_generated_decks(assets)
    belief_path = _write_belief_prior(assets)
    manifest_path = _write_manifest(assets, opponent_components)
    config_path = _write_pool_config(assets, opponent_components)
    readme_path = _write_readme(replay_decks, assets)
    _validate_outputs(assets, belief_path, opponent_components)
    observed_signatures = sum(asset.observed_games > 0 for asset in assets)
    print(
        f"wrote {observed_signatures} recent signatures from "
        f"{len(replay_decks)} public games"
    )
    print(f"belief: {belief_path.relative_to(_REPOSITORY_ROOT)}")
    print(f"manifest: {manifest_path.relative_to(_REPOSITORY_ROOT)}")
    print(f"deck pool: {config_path.relative_to(_REPOSITORY_ROOT)}")
    print(f"documentation: {readme_path.relative_to(_REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
