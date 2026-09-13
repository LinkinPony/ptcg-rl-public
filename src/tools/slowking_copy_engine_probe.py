"""Verify Slowking's copied payload effects against the bundled engine.

This is an offline parity gate for the small damage/KO thresholds used by the
fast scripted opponent. The engine still executes every effect online.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ptcg_rl.actions.selection import is_legal_action, random_legal_action
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.constants import AreaType, LogType, OptionType, SelectContext
from ptcg_rl.engine.runtime import to_engine_observation
from ptcg_rl.engine.session import BattleSession
from ptcg_rl.opponents.slowking_copy import view
from ptcg_rl.opponents.slowking_copy_v2 import (
    DECK,
    build_slowking_copy_agent,
    cards,
    tactics,
)

_PAYLOAD_ATTACKS = frozenset({cards.GUTSY_SWING, cards.TRIFROST, cards.DESTINED_FIGHT})
_PSYCHIC_TELEPATH_TARGETS = frozenset(
    {
        cards.SLOWPOKE,
        cards.SLOWKING,
        cards.SMOOCHUM,
        cards.LATIAS_EX,
    }
)
_REQUIRED_PARITY_CASES = frozenset(
    {
        "counter_gain_behind",
        "counter_gain_not_behind",
        "one_psychic_without_counter_gain",
        "telepath_psychic_target",
        "telepath_non_psychic_target",
    }
)


def run_probe(
    opponent_deck: Sequence[int],
    *,
    max_games: int = 128,
    max_steps_per_game: int = 1_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run bounded real battles until all three copied effects are verified."""
    verified: set[int] = set()
    parity_verified: set[str] = set()
    parity_observations: Counter[str] = Counter()
    attack_counts = dict.fromkeys(sorted(_PAYLOAD_ATTACKS), 0)
    games_run = 0

    for game_index in range(max_games):
        games_run += 1
        agent = build_slowking_copy_agent()
        agent.begin_game(player_index=0, own_deck=DECK)
        opponent_rng = random.Random(seed + game_index)
        pending_trifrost = False

        with BattleSession(DECK, opponent_deck) as battle:
            observation = battle.observation_dict
            for _ in range(max_steps_per_game):
                pending_trifrost = _inspect_logs(
                    observation.get("logs", ()),
                    attack_counts=attack_counts,
                    verified=verified,
                    pending_trifrost=pending_trifrost,
                )
                if int(observation["current"]["result"]) >= 0:
                    break

                player_index = int(observation["current"]["yourIndex"])
                select = observation["select"]
                if player_index == 0:
                    _observe_seek_cost_parity(
                        observation,
                        verified=parity_verified,
                        observations=parity_observations,
                    )
                    action = tuple(agent.act(observation))
                    action, telepath_case = _telepath_parity_action(
                        observation,
                        fallback=action,
                        verified=parity_verified,
                    )
                else:
                    action = random_legal_action(select, rng=opponent_rng)
                    telepath_case = None
                if not is_legal_action(select, action):
                    raise AssertionError(
                        f"illegal action during engine probe: {action!r}"
                    )
                trifrost_target_prompt = (
                    player_index == 0
                    and pending_trifrost
                    and int(select["context"])
                    in {
                        int(SelectContext.DAMAGE),
                        int(SelectContext.EFFECT_TARGET),
                    }
                )
                observation = battle.select(action)
                if telepath_case is not None:
                    _verify_telepath_result(
                        observation,
                        parity_case=telepath_case,
                    )
                    parity_verified.add(telepath_case)
                    parity_observations[telepath_case] += 1
                if trifrost_target_prompt:
                    damage_values = _opponent_hp_changes(observation.get("logs", ()))
                    if len(damage_values) == len(action) and all(
                        value == -110 for value in damage_values
                    ):
                        verified.add(cards.TRIFROST)
                    pending_trifrost = False
        if (
            verified == _PAYLOAD_ATTACKS
            and parity_verified >= _REQUIRED_PARITY_CASES
        ):
            break

    missing = sorted(_PAYLOAD_ATTACKS - verified)
    missing_parity = sorted(_REQUIRED_PARITY_CASES - parity_verified)
    if missing or missing_parity:
        raise AssertionError(
            "Slowking copy engine parity evidence was incomplete after "
            f"{games_run} games; missing_attacks={missing}, "
            f"missing_cases={missing_parity}, attack_counts={attack_counts}, "
            f"parity_observations={dict(parity_observations)}"
        )
    return {
        "engine_source_of_truth": True,
        "games_run": games_run,
        "verified_attack_ids": sorted(verified),
        "attack_counts": attack_counts,
        "verified_parity_cases": sorted(parity_verified),
        "parity_observations": dict(sorted(parity_observations.items())),
        "checks": {
            str(cards.GUTSY_SWING): "opponent HP_CHANGE is -250",
            str(cards.TRIFROST): "each selected opponent target HP_CHANGE is -110",
            str(cards.DESTINED_FIGHT): (
                "both Active Pokemon move from ACTIVE to DISCARD"
            ),
            "counter_gain": (
                "one Psychic enables Seek only while Counter Gain's Prize "
                "condition is active"
            ),
            "telepath": (
                "effect.id=19 TO_BENCH follows Psychic attachment only"
            ),
        },
    }


def _observe_seek_cost_parity(
    observation: Mapping[str, Any],
    *,
    verified: set[str],
    observations: Counter[str],
) -> None:
    """Compare the fast one-Energy readiness predicate with engine legality."""
    engine_observation = to_engine_observation(observation)
    select = engine_observation.select
    if select is None or int(select.context) != int(SelectContext.MAIN):
        return
    active = view.active(engine_observation, 0)
    if view.card_id(active) != cards.SLOWKING:
        return
    energies = tuple(int(value) for value in getattr(active, "energies", ()) or ())
    if energies != (cards.BASIC_PSYCHIC,):
        return

    has_counter_gain = cards.COUNTER_GAIN in view.card_ids(
        view.as_sequence(getattr(active, "tools", ()))
    )
    own_prizes = len(
        view.as_sequence(getattr(view.player(engine_observation, 0), "prize", ()))
    )
    opponent_prizes = len(
        view.as_sequence(getattr(view.player(engine_observation, 1), "prize", ()))
    )
    if has_counter_gain:
        case = (
            "counter_gain_behind"
            if own_prizes > opponent_prizes
            else "counter_gain_not_behind"
        )
        expected = own_prizes > opponent_prizes
    else:
        case = "one_psychic_without_counter_gain"
        expected = False

    engine_seek_legal = any(
        int(getattr(option, "type", -1)) == int(OptionType.ATTACK)
        and int(getattr(option, "attackId", -1)) == cards.SEEK_INSPIRATION
        for option in select.option
    )
    fast_ready = tactics.seek_cost_ready(engine_observation, 0, active)
    if engine_seek_legal != expected or fast_ready != engine_seek_legal:
        raise AssertionError(
            "Seek cost parity mismatch: "
            f"case={case}, prizes={own_prizes}:{opponent_prizes}, "
            f"engine_seek_legal={engine_seek_legal}, fast_ready={fast_ready}"
        )
    verified.add(case)
    observations[case] += 1


def _telepath_parity_action(
    observation: Mapping[str, Any],
    *,
    fallback: tuple[int, ...],
    verified: set[str],
) -> tuple[tuple[int, ...], str | None]:
    """Select one positive and one negative real Telepath trigger case."""
    engine_observation = to_engine_observation(observation)
    select = engine_observation.select
    if select is None or int(select.context) != int(SelectContext.MAIN):
        return fallback, None
    targets: list[tuple[int, int | None]] = []
    for index, option in enumerate(select.option):
        if (
            int(getattr(option, "type", -1)) == int(OptionType.ATTACH)
            and view.option_card_id(engine_observation, 0, option)
            == cards.TELEPATH_PSYCHIC_ENERGY
        ):
            targets.append(
                (
                    index,
                    view.card_id(view.option_pokemon(engine_observation, 0, option)),
                )
            )

    for case in (
        "telepath_psychic_target",
        "telepath_non_psychic_target",
    ):
        if case in verified:
            continue
        if (
            case == "telepath_psychic_target"
            and tactics.bench_space(engine_observation, 0) <= 0
        ):
            continue
        for index, target_id in targets:
            is_psychic = target_id in _PSYCHIC_TELEPATH_TARGETS
            matches = (
                target_id is not None
                and (
                    is_psychic
                    if case == "telepath_psychic_target"
                    else not is_psychic
                )
            )
            if matches:
                return (index,), case
    return fallback, None


def _verify_telepath_result(
    observation: Mapping[str, Any],
    *,
    parity_case: str,
) -> None:
    """Assert whether the engine emitted Telepath's follow-up Bench search."""
    engine_observation = to_engine_observation(observation)
    select = engine_observation.select
    effect = getattr(select, "effect", None)
    effect_id = getattr(effect, "id", None)
    triggered = (
        select is not None
        and int(select.context) == int(SelectContext.TO_BENCH)
        and effect_id is not None
        and int(effect_id) == cards.TELEPATH_PSYCHIC_ENERGY
    )
    expected = parity_case == "telepath_psychic_target"
    if triggered != expected:
        raise AssertionError(
            "Telepath trigger parity mismatch: "
            f"case={parity_case}, triggered={triggered}"
        )


def _inspect_logs(
    logs: Sequence[Mapping[str, Any]],
    *,
    attack_counts: dict[int, int],
    verified: set[int],
    pending_trifrost: bool,
) -> bool:
    attack_ids = {
        int(log.get("attackId", -1))
        for log in logs
        if int(log.get("type", -1)) == int(LogType.ATTACK)
        and int(log.get("playerIndex", -1)) == 0
    }
    for attack_id in attack_ids & _PAYLOAD_ATTACKS:
        attack_counts[attack_id] += 1
    if cards.GUTSY_SWING in attack_ids and -250 in _opponent_hp_changes(logs):
        verified.add(cards.GUTSY_SWING)
    if cards.DESTINED_FIGHT in attack_ids and _both_actives_discarded(logs):
        verified.add(cards.DESTINED_FIGHT)
    return pending_trifrost or cards.TRIFROST in attack_ids


def _opponent_hp_changes(logs: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(log.get("value", 0))
        for log in logs
        if int(log.get("type", -1)) == int(LogType.HP_CHANGE)
        and int(log.get("playerIndex", -1)) == 1
    ]


def _both_actives_discarded(logs: Sequence[Mapping[str, Any]]) -> bool:
    players = {
        int(log.get("playerIndex", -1))
        for log in logs
        if int(log.get("type", -1)) == int(LogType.MOVE_CARD)
        and int(log.get("fromArea", -1)) == int(AreaType.ACTIVE)
        and int(log.get("toArea", -1)) == int(AreaType.DISCARD)
    }
    return players == {0, 1}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opponent-deck",
        type=Path,
        default=Path("data/sample_submission/deck.csv"),
    )
    parser.add_argument("--max-games", type=int, default=128)
    parser.add_argument("--max-steps-per-game", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the command-line engine parity probe."""
    args = _parse_args()
    if args.max_games <= 0 or args.max_steps_per_game <= 0:
        raise ValueError("probe limits must be positive")
    opponent_deck = records.read_deck(records.repo_path(args.opponent_deck))
    result = run_probe(
        opponent_deck,
        max_games=args.max_games,
        max_steps_per_game=args.max_steps_per_game,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
