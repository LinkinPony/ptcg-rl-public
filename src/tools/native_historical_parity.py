"""Audit native-column historical inference against the JSON compatibility path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.policy_inputs import build_canonical_policy_input
from ptcg_rl.belief.public_catalog import (
    PublicDeckCatalog,
    PublicDeckCatalogConfig,
)
from ptcg_rl.context import (
    ContextBeliefTracker,
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.data.kaggle_deck.records import read_deck
from ptcg_rl.decks import canonicalize_deck
from ptcg_rl.engine.native_public_context import NativePublicContextTracker
from ptcg_rl.engine.native_training import (
    NativeTrainingBatchView,
    NativeTrainingLane,
    NativeTrainingOutputBuffer,
)
from ptcg_rl.model import collate_encoded_options, collate_state_tokens
from ptcg_rl.rl.native_historical_inference import NativeHistoricalPolicyPool
from ptcg_rl.rl.native_legacy_belief import (
    append_legacy_belief_tokens,
    legacy_known_counts,
)
from ptcg_rl.rl.native_policy_options import encode_native_option_batch
from ptcg_rl.rl.native_policy_state import encode_native_state_batch
from ptcg_rl.rl.stateless_training_config import StatelessHistoricalAnchorConfig

_FINGERPRINT_PLACEHOLDER = "0" * 64
_SELECTION_ADVANCE_PREFIX = "CG_TRAIN_SELECTION_ADVANCE_COUNT "


def main() -> None:
    """Run one artifact-bound, identical-trace parity audit."""
    arguments = _arguments()
    deck = canonicalize_deck(read_deck(arguments.deck))
    checkpoint_sha256 = _sha256(arguments.checkpoint)
    belief_sha256 = _sha256(arguments.belief_summary)
    if (
        arguments.checkpoint_sha256 is not None
        and checkpoint_sha256 != arguments.checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 differs from the requested artifact")
    if (
        arguments.belief_summary_sha256 is not None
        and belief_sha256 != arguments.belief_summary_sha256
    ):
        raise ValueError("belief summary SHA-256 differs from the requested artifact")

    belief_config = OpponentBeliefFeatureConfig(
        enabled=True,
        deck_signature_summary_path=arguments.belief_summary,
        deck_signature_summary_sha256=belief_sha256,
    )
    catalog = PublicDeckCatalog.from_config(
        PublicDeckCatalogConfig(
            manifest_path=arguments.public_catalog_manifest,
            catalog_fingerprint=arguments.public_catalog_fingerprint,
        )
    )
    resource = _resource(
        arguments,
        checkpoint_sha256=checkpoint_sha256,
        belief_sha256=belief_sha256,
        deck_digest=deck.deck_digest,
    )
    native_pool = NativeHistoricalPolicyPool(
        {arguments.member_id: (resource, deck.card_ids)}
    )
    reference_policy = CheckpointPolicy(
        arguments.checkpoint,
        device=arguments.device,
        own_deck=deck.card_ids,
    )
    reference_tracker = ContextBeliefTracker(belief=belief_config)
    reference_tracker.begin_game(player_index=0, own_deck=deck.card_ids)
    native_tracker = NativePublicContextTracker(
        slot_capacity=1,
        catalog=catalog,
        input_contract_fingerprint=arguments.input_contract_fingerprint,
    )
    native_belief = OpponentBeliefFeatureProducer.from_config(belief_config)
    observations = _reference_trace(
        arguments.parity_probe,
        seed=arguments.seed,
        maximum_steps=arguments.maximum_engine_steps,
        deck=deck.card_ids,
    )
    decks = np.asarray([[deck.card_ids, deck.card_ids]], dtype=np.int32)
    first = NativeTrainingOutputBuffer(slot_capacity=1, option_capacity=256)
    second = NativeTrainingOutputBuffer(slot_capacity=1, option_capacity=256)
    compared = 0
    action_rows = 0
    forced_rows = 0
    state_fields_checked = 0
    option_fields_checked = 0
    trace_rows_checked = 0
    selection_advances_checked = 0

    with NativeTrainingLane(1, library_path=arguments.library) as lane:
        view = lane.reset(
            decks,
            np.asarray([arguments.seed], dtype=np.uint32),
            output=first,
        )
        context = native_tracker.consume_reset(view, decks)
        for step, (observation, expected_advances) in enumerate(observations):
            _require_aligned(view, observation, step=step)
            actual_advances = int(view.selection_advance_count[0])
            if actual_advances != expected_advances:
                raise RuntimeError(
                    "selection-advance trace parity failed at step "
                    f"{step}: native={actual_advances}, "
                    f"reference={expected_advances}"
                )
            trace_rows_checked += 1
            selection_advances_checked += actual_advances
            if int(view.select_player[0]) == 0 and view.option_count > 0:
                adapted = reference_tracker.observation_with_context(observation)
                known = native_tracker.known_opponent_batch(
                    np.asarray([0], dtype=np.uint32),
                    np.asarray([0], dtype=np.int32),
                )
                native_states, lookup = encode_native_state_batch(
                    view,
                    context,
                    device="cpu",
                )
                native_options = encode_native_option_batch(
                    view,
                    lookup,
                    device="cpu",
                )
                known_counts = legacy_known_counts(view, known, row=0)
                native_states = append_legacy_belief_tokens(
                    native_states,
                    (native_belief.features_from_known_counts(known_counts),),
                )
                policy_input = build_canonical_policy_input(adapted)
                if policy_input is None:
                    raise RuntimeError("reference historical input has no options")
                reference_states = collate_state_tokens(
                    (policy_input.state,),
                    device="cpu",
                )
                reference_options = collate_encoded_options(
                    (policy_input.options,),
                    min_counts=(policy_input.min_count,),
                    max_counts=(policy_input.max_count,),
                    device="cpu",
                )
                state_fields_checked += _assert_tensor_fields_equal(
                    native_states,
                    reference_states,
                    label=f"state step {step}",
                )
                option_fields_checked += _assert_tensor_fields_equal(
                    native_options,
                    reference_options,
                    label=f"options step {step}",
                )

                forced = _forced_action(view)
                reference_action = (
                    forced
                    if forced is not None
                    else reference_policy.select_action(adapted)
                )
                (native_action,) = native_pool.act_many(
                    view,
                    context,
                    known,
                    member_ids=(arguments.member_id,),
                    forced_actions=(forced,),
                )
                if native_action != reference_action:
                    raise RuntimeError(
                        "decoded action parity failed at step "
                        f"{step}: native={native_action}, "
                        f"reference={reference_action}"
                    )
                compared += 1
                forced_rows += int(forced is not None)
                action_rows += int(forced is None)
                if compared >= arguments.comparison_rows:
                    break

            if int(view.result[0]) >= 0:
                break
            offsets, selected = _minimum_action(view)
            view = lane.step(
                np.asarray([0], dtype=np.uint32),
                offsets,
                selected,
                output=second if step % 2 == 0 else first,
            )
            context = native_tracker.consume_step(view)

    if compared < arguments.comparison_rows:
        raise RuntimeError(
            f"trace exposed only {compared} anchor-seat decisions; "
            f"requested {arguments.comparison_rows}"
        )
    report = {
        "format": "native-historical-inference-parity-v2",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "member_id": arguments.member_id,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "belief_summary": str(arguments.belief_summary.resolve()),
        "belief_summary_sha256": belief_sha256,
        "deck": str(arguments.deck.resolve()),
        "deck_digest": deck.deck_digest,
        "seed": arguments.seed,
        "comparison_rows": compared,
        "decoded_action_rows": action_rows,
        "forced_action_rows": forced_rows,
        "state_tensor_fields_checked": state_fields_checked,
        "option_tensor_fields_checked": option_fields_checked,
        "tensor_parity": "bitwise_equal",
        "decoded_action_parity": "exact",
        "selection_advance_trace_parity": "exact",
        "selection_advance_trace_rows_checked": trace_rows_checked,
        "selection_advances_checked": selection_advances_checked,
        "native_loaded_checkpoint_artifacts": native_pool.loaded_artifact_count,
        "trace_policy": "deterministic minimum legal actions",
    }
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--belief-summary", type=Path, required=True)
    parser.add_argument("--belief-summary-sha256")
    parser.add_argument("--public-catalog-manifest", type=Path, required=True)
    parser.add_argument("--public-catalog-fingerprint", required=True)
    parser.add_argument("--input-contract-fingerprint", required=True)
    parser.add_argument(
        "--exact-registry-fingerprint",
        default=_FINGERPRINT_PLACEHOLDER,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--library", type=Path)
    parser.add_argument(
        "--parity-probe",
        type=Path,
        default=Path("tmp/cg_train_parity_probe"),
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--maximum-engine-steps", type=int, default=512)
    parser.add_argument("--comparison-rows", type=int, default=16)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.maximum_engine_steps <= 0 or arguments.comparison_rows <= 0:
        parser.error("step and comparison limits must be positive")
    if not 0 <= arguments.seed <= np.iinfo(np.uint32).max:
        parser.error("--seed must fit uint32")
    return arguments


def _resource(
    arguments: argparse.Namespace,
    *,
    checkpoint_sha256: str,
    belief_sha256: str,
    deck_digest: str,
) -> StatelessHistoricalAnchorConfig:
    return StatelessHistoricalAnchorConfig(
        member_id=arguments.member_id,
        snapshot_id=f"{arguments.member_id}-parity",
        checkpoint_path=arguments.checkpoint,
        checkpoint_size_bytes=arguments.checkpoint.stat().st_size,
        checkpoint_sha256=checkpoint_sha256,
        pilot_artifact_fingerprint=_FINGERPRINT_PLACEHOLDER,
        bundle_fingerprint=_FINGERPRINT_PLACEHOLDER,
        exact_deck_path=arguments.deck,
        exact_deck_digest=deck_digest,
        input_contract_fingerprint=arguments.input_contract_fingerprint,
        exact_registry_fingerprint=arguments.exact_registry_fingerprint,
        belief_summary_path=arguments.belief_summary,
        belief_summary_sha256=belief_sha256,
        device=arguments.device,
    )


def _reference_trace(
    executable: Path,
    *,
    seed: int,
    maximum_steps: int,
    deck: tuple[int, ...],
) -> tuple[tuple[dict[str, Any], int], ...]:
    deck_input = " ".join(str(card_id) for card_id in (*deck, *deck)) + "\n"
    completed = subprocess.run(
        (str(executable.resolve()), str(seed), str(maximum_steps)),
        input=deck_input,
        text=True,
        capture_output=True,
        check=True,
    )
    observations = tuple(json.loads(line) for line in completed.stdout.splitlines())
    counts = _selection_advance_counts(completed.stderr)
    if len(observations) != len(counts):
        raise RuntimeError(
            "debug trace emitted different observation and selection-advance row counts"
        )
    return tuple(zip(observations, counts, strict=True))


def _selection_advance_counts(stderr: str) -> tuple[int, ...]:
    counts: list[int] = []
    unexpected: list[str] = []
    for line in stderr.splitlines():
        if line.startswith(_SELECTION_ADVANCE_PREFIX):
            raw_count = line.removeprefix(_SELECTION_ADVANCE_PREFIX)
            try:
                count = int(raw_count)
            except ValueError as error:
                raise RuntimeError(
                    "debug trace emitted an invalid selection-advance count"
                ) from error
            if not 0 <= count <= np.iinfo(np.uint32).max:
                raise RuntimeError(
                    "debug trace selection-advance count is outside uint32"
                )
            counts.append(count)
        elif line:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "debug trace wrote unexpected stderr: " + "\n".join(unexpected)
        )
    return tuple(counts)


def _assert_tensor_fields_equal(left: Any, right: Any, *, label: str) -> int:
    checked = 0
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, Tensor):
            if not isinstance(right_value, Tensor) or not torch.equal(
                left_value,
                right_value,
            ):
                maximum = _maximum_difference(left_value, right_value)
                mismatch = _first_mismatch(left_value, right_value)
                raise RuntimeError(
                    f"{label}.{field.name} differs; max_abs={maximum}; first={mismatch}"
                )
            checked += 1
    return checked


def _maximum_difference(left: Any, right: Any) -> float | str:
    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        return "non_tensor"
    if left.shape != right.shape:
        return f"shape:{tuple(left.shape)}!={tuple(right.shape)}"
    if left.numel() == 0:
        return 0.0
    return float((left.to(torch.float64) - right.to(torch.float64)).abs().max().item())


def _first_mismatch(left: Any, right: Any) -> str:
    if (
        not isinstance(left, Tensor)
        or not isinstance(right, Tensor)
        or left.shape != right.shape
    ):
        return "unavailable"
    indices = torch.nonzero(left != right, as_tuple=False)
    if not int(indices.shape[0]):
        return "none"
    index = tuple(int(value) for value in indices[0].tolist())
    return f"index={index},native={left[index].item()},reference={right[index].item()}"


def _require_aligned(
    view: NativeTrainingBatchView,
    observation: dict[str, Any],
    *,
    step: int,
) -> None:
    if view.batch_size != 1 or int(view.error[0]) != 0:
        raise RuntimeError(f"native trace failed at step {step}")
    current = observation.get("current")
    if not isinstance(current, dict):
        raise RuntimeError(f"reference trace has no current state at step {step}")
    if int(current.get("yourIndex", -1)) != int(view.select_player[0]):
        raise RuntimeError(f"acting perspective differs at step {step}")


def _forced_action(
    view: NativeTrainingBatchView,
) -> tuple[int, ...] | None:
    option_count = int(view.option_offsets[1] - view.option_offsets[0])
    minimum = min(option_count, max(0, int(view.select_min[0])))
    maximum = min(option_count, max(minimum, int(view.select_max[0])))
    if maximum == 0:
        return ()
    if option_count == 1 and minimum == 1 and maximum == 1:
        return (0,)
    return None


def _minimum_action(
    view: NativeTrainingBatchView,
) -> tuple[np.ndarray, np.ndarray]:
    option_count = int(view.option_offsets[1] - view.option_offsets[0])
    length = min(option_count, max(0, int(view.select_min[0])))
    return (
        np.asarray([0, length], dtype=np.uint32),
        np.arange(length, dtype=np.int32),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
