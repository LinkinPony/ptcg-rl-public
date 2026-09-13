"""Compare native-column mixed75 decisions with the immutable legacy pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingLane,
    NativeTrainingOutputBuffer,
)
from ptcg_rl.opponents import build_opponent, opponent_registry
from ptcg_rl.rl.native_scripted_mixed75 import NativeMixed75Policy

_DECK_ENV = "POKEMON_TCG_DECK_PATH"
_STRATEGY_ENV = "POKEMON_TCG_STRATEGY_ID"
_GENERIC_STRATEGY = "generic_water_mcts"
_SOURCE_FILES = (
    Path("src/ptcg_rl/engine/native_training.py"),
    Path("src/ptcg_rl/rl/native_scripted_catalog.py"),
    Path("src/ptcg_rl/rl/native_scripted_heuristic.py"),
    Path("src/ptcg_rl/rl/native_scripted_heuristic_board.py"),
    Path("src/ptcg_rl/rl/native_scripted_mixed75.py"),
    Path("src/ptcg_rl/rl/native_scripted_predicates.py"),
    Path("src/ptcg_rl/rl/native_scripted_state.py"),
    Path("src/ptcg_rl/opponents/builtin.py"),
    Path("src/ptcg_rl/opponents/spec.py"),
    Path("src/ptcg_rl/opponents/third_party.py"),
    Path("third_party/pokemon-tcg-ai-battle/src/heuristic/agent.py"),
)


def main() -> None:
    """Run exact decision parity over deterministic source-engine traces."""
    args = _parse_args()
    report = run_parity(
        scripted_deck_path=args.scripted_deck,
        candidate_deck_paths=tuple(args.candidate_deck),
        static_features_path=args.static_features,
        native_library_path=args.native_library,
        reference_probe_path=args.reference_probe,
        engine_seeds=_parse_seeds(args.engine_seeds),
        policy_seed_base=args.policy_seed_base,
        scripted_seat=args.scripted_seat,
        maximum_steps=args.maximum_steps,
        require_coverage=not args.allow_incomplete_coverage,
        required_skill_card_ids=tuple(args.require_skill_card_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def run_parity(
    *,
    scripted_deck_path: Path,
    candidate_deck_paths: Sequence[Path],
    static_features_path: Path,
    native_library_path: Path,
    reference_probe_path: Path,
    engine_seeds: Sequence[int],
    policy_seed_base: int,
    scripted_seat: int,
    maximum_steps: int,
    require_coverage: bool,
    required_skill_card_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Return artifact-bound exact parity evidence or raise on first drift."""
    if not candidate_deck_paths:
        raise ValueError("parity requires at least one candidate deck")
    if not engine_seeds:
        raise ValueError("parity requires at least one engine seed")
    if scripted_seat not in (0, 1):
        raise ValueError("scripted seat must be zero or one")
    if maximum_steps <= 0:
        raise ValueError("maximum steps must be positive")
    if any(card_id <= 0 for card_id in required_skill_card_ids):
        raise ValueError("required skill card IDs must be positive")
    for path in (
        scripted_deck_path,
        static_features_path,
        native_library_path,
        reference_probe_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    scripted = _read_deck(scripted_deck_path)
    candidates = tuple(_read_deck(path) for path in candidate_deck_paths)
    static_features = np.load(static_features_path, mmap_mode="r")
    totals: Counter[str] = Counter()
    traces: list[dict[str, Any]] = []
    policy_seconds = 0.0

    with _artifact_environment(scripted_deck_path):
        for candidate_index, (candidate_path, candidate) in enumerate(
            zip(candidate_deck_paths, candidates, strict=True)
        ):
            for seed_index, engine_seed in enumerate(engine_seeds):
                policy_seed = (
                    int(policy_seed_base)
                    + candidate_index * len(engine_seeds)
                    + seed_index
                )
                trace, elapsed = _compare_trace(
                    candidate=candidate,
                    scripted=scripted,
                    candidate_path=candidate_path,
                    engine_seed=int(engine_seed),
                    policy_seed=policy_seed,
                    scripted_seat=scripted_seat,
                    maximum_steps=maximum_steps,
                    static_features=static_features,
                    native_library_path=native_library_path,
                    reference_probe_path=reference_probe_path,
                )
                traces.append(trace)
                policy_seconds += elapsed
                totals.update(trace["coverage"])

    required = (
        "forced",
        "heuristic",
        "random",
        "count_prompt",
        "multi_select_prompt",
    )
    missing = tuple(name for name in required if totals[name] <= 0)
    missing_skill_cards = tuple(
        card_id
        for card_id in required_skill_card_ids
        if totals[f"skill_card_{card_id}_option"] <= 0
    )
    if require_coverage and missing:
        raise RuntimeError(
            "native scripted parity coverage is incomplete: "
            + ", ".join(missing)
        )
    if missing_skill_cards:
        raise RuntimeError(
            "native scripted parity did not observe required SKILL cards: "
            + ", ".join(str(card_id) for card_id in missing_skill_cards)
        )
    compared = totals["compared_decisions"]
    root = Path.cwd()
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "result": "exact_parity",
        "mismatch_count": 0,
        "trace_count": len(traces),
        "compared_decisions": compared,
        "native_policy_seconds": policy_seconds,
        "native_policy_decisions_per_second": (
            compared / policy_seconds if policy_seconds > 0.0 else 0.0
        ),
        "coverage": dict(sorted(totals.items())),
        "coverage_missing": list(missing),
        "observed_skill_card_ids": sorted(
            {
                card_id
                for trace in traces
                for card_id in trace["skill_card_ids"]
            }
        ),
        "required_skill_card_ids": list(required_skill_card_ids),
        "scripted_seat": scripted_seat,
        "maximum_steps": maximum_steps,
        "scripted_deck": _deck_evidence(scripted_deck_path, scripted),
        "candidate_decks": [
            _deck_evidence(path, deck)
            for path, deck in zip(
                candidate_deck_paths,
                candidates,
                strict=True,
            )
        ],
        "engine_seeds": [int(seed) for seed in engine_seeds],
        "policy_seed_base": int(policy_seed_base),
        "static_features": _file_evidence(static_features_path),
        "native_library": _file_evidence(native_library_path),
        "reference_probe": _file_evidence(reference_probe_path),
        "source_files": {
            str(path): _sha256(root / path) for path in _SOURCE_FILES
        },
        "traces": traces,
        "scope": (
            "Legacy mixed75 consumes canonical ToJsonApi observations from the "
            "debug-only source-engine trace. The candidate consumes the "
            "aligned no-JSON native SoA/CSR view. Both are initialized with "
            "the same per-game policy seed and compared only on the scripted "
            "seat; the game itself follows the same minimum-legal action "
            "trace on both engine instances."
        ),
    }


def _compare_trace(
    *,
    candidate: CanonicalDeck,
    scripted: CanonicalDeck,
    candidate_path: Path,
    engine_seed: int,
    policy_seed: int,
    scripted_seat: int,
    maximum_steps: int,
    static_features: np.ndarray,
    native_library_path: Path,
    reference_probe_path: Path,
) -> tuple[dict[str, Any], float]:
    deck_pair = (
        (scripted, candidate)
        if scripted_seat == 0
        else (candidate, scripted)
    )
    reference = _reference_trace(
        reference_probe_path,
        deck_pair=deck_pair,
        engine_seed=engine_seed,
        maximum_steps=maximum_steps,
    )
    legacy = build_opponent(opponent_registry()["mixed75"], seed=policy_seed)
    native = NativeMixed75Policy(
        scripted_deck=scripted.card_ids,
        static_features=static_features,
    )
    slots = np.asarray((0,), dtype=np.uint32)
    native.reset(slots, np.asarray((policy_seed,), dtype=np.uint32))
    output = NativeTrainingOutputBuffer(
        slot_capacity=1,
        option_capacity=512,
        visible_card_capacity=512,
        attachment_capacity=512,
        log_capacity=4096,
    )
    decks = np.asarray(
        ((deck_pair[0].card_ids, deck_pair[1].card_ids),),
        dtype=np.int32,
    )
    coverage: Counter[str] = Counter()
    skill_card_ids: set[int] = set()
    policy_seconds = 0.0
    terminal_seen = False

    with NativeTrainingLane(
        1,
        library_path=native_library_path,
    ) as lane:
        view = lane.reset(
            decks,
            np.asarray((engine_seed,), dtype=np.uint32),
            output=output,
            slots=slots,
        )
        for step, observation in enumerate(reference):
            try:
                _assert_select_alignment(view, observation, step=step)
            except AssertionError as exc:
                raise AssertionError(
                    "reference/native engine trace drift: "
                    f"candidate={candidate_path} engine_seed={engine_seed} "
                    f"step={step}: {exc}"
                ) from exc
            result = int(_mapping(_mapping(observation, "current"), "result"))
            if result >= 0:
                if int(view.status[0]) != 2:
                    raise AssertionError(
                        "reference/native terminal status diverged at "
                        f"step {step}"
                    )
                terminal_seen = True
                break
            current = _mapping(observation, "current")
            perspective = int(_mapping(current, "yourIndex"))
            select = _mapping(observation, "select")
            if perspective == scripted_seat:
                raw_options = _sequence(_mapping(select, "option"))
                prompt_skill_ids = tuple(
                    int(_mapping(_as_mapping(option), "cardId"))
                    for option in raw_options
                    if int(_mapping(_as_mapping(option), "type")) == 15
                )
                if prompt_skill_ids:
                    coverage["skill_prompt"] += 1
                    coverage["skill_options"] += len(prompt_skill_ids)
                    for card_id in prompt_skill_ids:
                        skill_card_ids.add(card_id)
                        coverage[f"skill_card_{card_id}_option"] += 1
                        if card_id not in scripted.card_ids:
                            coverage["foreign_skill_options"] += 1
                legacy_action = tuple(int(index) for index in legacy.act(observation))
                started = time.perf_counter()
                try:
                    native_batch = native.act_batch(view)
                except Exception as exc:
                    raise RuntimeError(
                        "native mixed75 policy failed: "
                        f"candidate={candidate_path} engine_seed={engine_seed} "
                        f"policy_seed={policy_seed} step={step} "
                        f"skill_card_ids={prompt_skill_ids}"
                    ) from exc
                policy_seconds += time.perf_counter() - started
                native_action = native_batch.action(0)
                branch = native_batch.branches[0]
                coverage["compared_decisions"] += 1
                coverage[branch] += 1
                if int(_mapping(select, "type")) == 8:
                    coverage["count_prompt"] += 1
                if int(_mapping(select, "maxCount")) > 1:
                    coverage["multi_select_prompt"] += 1
                if legacy_action != native_action:
                    raise AssertionError(
                        "mixed75 decision mismatch: "
                        f"candidate={candidate_path} engine_seed={engine_seed} "
                        f"policy_seed={policy_seed} step={step} "
                        f"branch={branch} legacy={legacy_action} "
                        f"native={native_action} "
                        f"type={_mapping(select, 'type')} "
                        f"context={_mapping(select, 'context')}"
                    )
            minimum = int(view.select_min[0])
            action = np.arange(minimum, dtype=np.int32)
            view = lane.step(
                slots,
                np.asarray((0, minimum), dtype=np.uint32),
                action,
                output=output,
            )
    if not terminal_seen and len(reference) < maximum_steps:
        raise AssertionError("reference/native trace ended without terminal state")
    return (
        {
            "candidate_deck_digest": candidate.deck_digest,
            "engine_seed": engine_seed,
            "policy_seed": policy_seed,
            "reference_rows": len(reference),
            "terminal_seen": terminal_seen,
            "coverage": dict(sorted(coverage.items())),
            "skill_card_ids": sorted(skill_card_ids),
        },
        policy_seconds,
    )


def _reference_trace(
    probe: Path,
    *,
    deck_pair: tuple[CanonicalDeck, CanonicalDeck],
    engine_seed: int,
    maximum_steps: int,
) -> tuple[dict[str, Any], ...]:
    payload = " ".join(
        str(card_id)
        for deck in deck_pair
        for card_id in deck.card_ids
    )
    completed = subprocess.run(
        (str(probe), str(engine_seed), str(maximum_steps)),
        input=payload + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "source-engine parity probe failed: "
            f"code={completed.returncode} stderr={completed.stderr.strip()}"
        )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        decoded = json.loads(line)
        if not isinstance(decoded, dict):
            raise TypeError("reference engine emitted a non-object observation")
        rows.append(decoded)
    if not rows:
        raise RuntimeError("reference engine emitted no observations")
    return tuple(rows)


def _assert_select_alignment(
    view: NativeTrainingBatchView,
    observation: Mapping[str, Any],
    *,
    step: int,
) -> None:
    current = _mapping(observation, "current")
    select = _mapping(observation, "select")
    result = int(_mapping(current, "result"))
    if result >= 0:
        terminal_scalars = (
            (int(view.select_player[0]), int(_mapping(current, "yourIndex"))),
            (int(view.result[0]), result),
            (int(view.turn[0]), int(_mapping(current, "turn"))),
        )
        if any(
            actual != expected for actual, expected in terminal_scalars
        ):
            raise AssertionError(
                f"reference/native terminal scalar drift at step {step}"
            )
        return
    scalars = (
        (int(view.select_player[0]), int(_mapping(current, "yourIndex"))),
        (int(view.select_type[0]) - 1, int(_mapping(select, "type"))),
        (int(view.select_context[0]) - 1, int(_mapping(select, "context"))),
        (int(view.select_min[0]), int(_mapping(select, "minCount"))),
        (int(view.select_max[0]), int(_mapping(select, "maxCount"))),
        (int(view.result[0]), result),
    )
    if any(actual != expected for actual, expected in scalars):
        raise AssertionError(f"reference/native select scalar drift at step {step}")
    options = _sequence(_mapping(select, "option"))
    start = int(view.option_offsets[0])
    stop = int(view.option_offsets[1])
    if stop - start != len(options):
        raise AssertionError(f"reference/native option count drift at step {step}")
    for local, raw_option in enumerate(options):
        option = _as_mapping(raw_option)
        absolute = start + local
        option_type = int(_mapping(option, "type"))
        if int(view.option_type[absolute]) != option_type:
            raise AssertionError(
                f"reference/native option type drift at step {step}"
            )
        expected = _option_params(option_type, option)
        actual = tuple(
            int(column[absolute]) for column in view.option_params
        )
        for parameter, value in enumerate(expected):
            if value is not None and actual[parameter] != value:
                raise AssertionError(
                    "reference/native option parameter drift at "
                    f"step {step}, option {local}, parameter {parameter}"
                )


def _option_params(
    option_type: int,
    option: Mapping[str, Any],
) -> tuple[int | None, ...]:
    values: list[int | None] = [None] * 5
    if option_type == 0:
        values[0] = int(_mapping(option, "number"))
    elif option_type in (3, 4, 5, 6):
        values[0] = int(_mapping(option, "area"))
        values[1] = int(_mapping(option, "index"))
        values[2] = int(_mapping(option, "playerIndex"))
        if option_type == 4:
            values[3] = int(_mapping(option, "toolIndex"))
        elif option_type in (5, 6):
            values[3] = int(_mapping(option, "energyIndex"))
            if option_type == 6:
                values[4] = int(_mapping(option, "count"))
    elif option_type == 7:
        values[0] = int(_mapping(option, "index"))
    elif option_type in (8, 9):
        values[0] = int(_mapping(option, "area"))
        values[1] = int(_mapping(option, "index"))
        values[2] = int(_mapping(option, "inPlayArea"))
        values[3] = int(_mapping(option, "inPlayIndex"))
    elif option_type in (10, 11):
        values[0] = int(_mapping(option, "area"))
        values[1] = int(_mapping(option, "index"))
    elif option_type == 13:
        values[0] = int(_mapping(option, "attackId"))
    elif option_type == 15:
        values[0] = int(_mapping(option, "cardId"))
        values[1] = int(_mapping(option, "serial"))
    elif option_type == 16:
        values[0] = int(_mapping(option, "specialConditionType"))
    return tuple(values)


def _read_deck(path: Path) -> CanonicalDeck:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return canonicalize_deck(values)


def _deck_evidence(path: Path, deck: CanonicalDeck) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "deck_digest": deck.deck_digest,
    }


def _file_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _artifact_environment(deck_path: Path) -> Iterator[None]:
    previous_deck = os.environ.get(_DECK_ENV)
    previous_strategy = os.environ.get(_STRATEGY_ENV)
    os.environ[_DECK_ENV] = str(deck_path.resolve())
    os.environ[_STRATEGY_ENV] = _GENERIC_STRATEGY
    try:
        yield
    finally:
        if previous_deck is None:
            os.environ.pop(_DECK_ENV, None)
        else:
            os.environ[_DECK_ENV] = previous_deck
        if previous_strategy is None:
            os.environ.pop(_STRATEGY_ENV, None)
        else:
            os.environ[_STRATEGY_ENV] = previous_strategy


def _mapping(value: Mapping[str, Any], key: str) -> Any:
    if key not in value:
        raise KeyError(f"reference observation is missing {key!r}")
    return value[key]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("reference observation value is not an object")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("reference observation value is not an array")
    return value


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in seeds):
        raise ValueError("engine seeds must fit uint32")
    return seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted-deck", type=Path, required=True)
    parser.add_argument(
        "--candidate-deck",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--static-features", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--engine-seeds", required=True)
    parser.add_argument("--policy-seed-base", type=int, required=True)
    parser.add_argument("--scripted-seat", type=int, choices=(0, 1), default=1)
    parser.add_argument("--maximum-steps", type=int, default=1000)
    parser.add_argument("--allow-incomplete-coverage", action="store_true")
    parser.add_argument(
        "--require-skill-card-id",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
