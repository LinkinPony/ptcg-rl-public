"""Summary-level provenance checks for paired gauntlet analysis."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

_IDENTITY_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_source_commit",
    "pair_manifest_sha256",
    "public_catalog_manifest_sha256",
)


def read_manifest(games_path: Path) -> dict[str, Any]:
    """Read the immutable campaign manifest adjacent to a games artifact."""
    path = games_path.parent / "manifest.json"
    if not path.is_file():
        raise ValueError(f"completed artifact is missing manifest.json: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return cast(dict[str, Any], raw)


def read_summary(games_path: Path) -> dict[str, Any]:
    """Read the completion summary adjacent to a games artifact."""
    path = games_path.parent / "summary.json"
    if not path.is_file():
        raise ValueError(f"completed artifact is missing summary.json: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"summary is not a JSON object: {path}")
    return cast(dict[str, Any], raw)


def validate_summaries(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    control_temperature: float | None,
    treatment_temperature: float | None,
    baseline_temperature: float,
) -> dict[str, Any]:
    """Validate common source identity and the single temperature contrast."""
    expected_format = "native_checkpoint_gauntlet_summary_v1"
    if (
        control.get("format") != expected_format
        or treatment.get("format") != expected_format
    ):
        raise ValueError(
            "both artifacts must be completed gauntlet summary v1 artifacts"
        )
    namespaces = (
        control.get("match_seed_namespace"),
        treatment.get("match_seed_namespace"),
    )
    if not all(isinstance(item, str) and item.strip() for item in namespaces):
        raise ValueError(
            "paired artifacts require a shared nonempty match_seed_namespace"
        )
    if namespaces[0] != namespaces[1]:
        raise ValueError("control and treatment match_seed_namespace values differ")
    campaign_fingerprints = (
        control.get("campaign_fingerprint"),
        treatment.get("campaign_fingerprint"),
    )
    if not all(
        isinstance(item, str) and item.strip() for item in campaign_fingerprints
    ):
        raise ValueError("paired campaign fingerprints are missing or invalid")
    if campaign_fingerprints[0] == campaign_fingerprints[1]:
        raise ValueError("temperature arms must have distinct campaign fingerprints")

    control_candidate = participant(control, "candidate")
    treatment_candidate = participant(treatment, "candidate")
    control_baseline = participant(control, "baseline")
    treatment_baseline = participant(treatment, "baseline")
    actual_control = _temperature(control_candidate, "control candidate")
    actual_treatment = _temperature(treatment_candidate, "treatment candidate")
    wanted_control = 0.0 if control_temperature is None else control_temperature
    wanted_treatment = 1.0 if treatment_temperature is None else treatment_temperature
    _require_expected_temperature(actual_control, wanted_control, "control candidate")
    _require_expected_temperature(
        actual_treatment, wanted_treatment, "treatment candidate"
    )
    if math.isclose(actual_control, actual_treatment, abs_tol=1e-12):
        raise ValueError("candidate control and treatment temperatures must differ")

    baseline_temperatures = (
        _temperature(control_baseline, "control baseline"),
        _temperature(treatment_baseline, "treatment baseline"),
    )
    if not math.isclose(*baseline_temperatures, abs_tol=1e-12):
        raise ValueError("baseline policy temperature differs between arms")
    _require_expected_temperature(
        baseline_temperatures[0], baseline_temperature, "baseline"
    )
    participants = (
        control_candidate,
        treatment_candidate,
        control_baseline,
        treatment_baseline,
    )
    identities: dict[str, str] = {}
    for field in _IDENTITY_FIELDS:
        values = {str(item.get(field, "")) for item in participants}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"checkpoint provenance field {field} is not identical")
        identities[field] = next(iter(values))
    runner_source = control.get("runner_source_commit")
    if not isinstance(runner_source, str) or not runner_source:
        raise ValueError("runner source commit is missing or invalid")
    if runner_source != treatment.get("runner_source_commit"):
        raise ValueError("runner source commit differs between paired arms")
    return {
        "match_seed_namespace": namespaces[0],
        "control_candidate_temperature": actual_control,
        "treatment_candidate_temperature": actual_treatment,
        "baseline_temperature": baseline_temperatures[0],
        "runner_source_commit": runner_source,
        **identities,
        "control_campaign_fingerprint": campaign_fingerprints[0],
        "treatment_campaign_fingerprint": campaign_fingerprints[1],
    }


def validate_manifests(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    control_summary: dict[str, Any],
    treatment_summary: dict[str, Any],
) -> dict[str, Any]:
    """Require the campaign inputs to differ only in candidate temperature identity."""
    expected_format = "native_checkpoint_gauntlet_v1"
    for arm, manifest, summary in (
        ("control", control, control_summary),
        ("treatment", treatment, treatment_summary),
    ):
        if manifest.get("format") != expected_format:
            raise ValueError(f"{arm} campaign manifest format is unsupported")
        if campaign_manifest_fingerprint(manifest) != manifest.get(
            "campaign_fingerprint"
        ):
            raise ValueError(f"{arm} campaign manifest fingerprint is invalid")
        if manifest.get("campaign_fingerprint") != summary.get("campaign_fingerprint"):
            raise ValueError(f"{arm} campaign fingerprint differs from summary")
        if manifest.get("runner_source_commit") != summary.get("runner_source_commit"):
            raise ValueError(
                f"{arm} runner source differs between manifest and summary"
            )
        for role in ("candidate", "baseline"):
            _validate_manifest_participant(
                _mapping(manifest.get(role), f"{arm} manifest {role}"),
                participant(summary, role),
                label=f"{arm} {role}",
            )

    control_schedule = _mapping(control.get("schedule"), "control schedule")
    treatment_schedule = _mapping(treatment.get("schedule"), "treatment schedule")
    if control_schedule != treatment_schedule:
        raise ValueError("paired campaign schedules differ")
    namespace = control_summary.get("match_seed_namespace")
    if control_schedule.get("match_seed_namespace") != namespace:
        raise ValueError("campaign schedule namespace differs from summary")
    if control_schedule.get("total_games") != control_summary.get("games"):
        raise ValueError("campaign scheduled game count differs from summary")
    seed = control_schedule.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("campaign schedule seed is invalid")
    for field in ("cross_roster", "mirrored_seats", "full_cell_coverage"):
        if control_schedule.get(field) is not True:
            raise ValueError(f"campaign schedule requires {field}=true")

    control_native = _mapping(control.get("native_library"), "control native library")
    treatment_native = _mapping(
        treatment.get("native_library"), "treatment native library"
    )
    if control_native != treatment_native:
        raise ValueError("paired campaigns use different native libraries")
    native_sha256 = _required_string(control_native.get("sha256"), "native library sha")
    control_belief = _required_string(
        control.get("belief_fingerprint"), "control belief fingerprint"
    )
    treatment_belief = _required_string(
        treatment.get("belief_fingerprint"), "treatment belief fingerprint"
    )
    if control_belief != treatment_belief:
        raise ValueError("paired campaigns use different belief inputs")
    control_execution = _mapping(control.get("execution"), "control execution")
    treatment_execution = _mapping(treatment.get("execution"), "treatment execution")
    if control_execution != treatment_execution:
        raise ValueError("paired campaign execution configurations differ")
    maximum_engine_steps = control_execution.get("maximum_engine_steps")
    if (
        isinstance(maximum_engine_steps, bool)
        or not isinstance(maximum_engine_steps, int)
        or maximum_engine_steps <= 0
    ):
        raise ValueError("campaign maximum_engine_steps is invalid")

    control_baseline = _mapping(control.get("baseline"), "control baseline")
    treatment_baseline = _mapping(treatment.get("baseline"), "treatment baseline")
    if control_baseline != treatment_baseline:
        raise ValueError("paired campaign baseline participants differ")
    control_candidate = dict(_mapping(control.get("candidate"), "control candidate"))
    treatment_candidate = dict(
        _mapping(treatment.get("candidate"), "treatment candidate")
    )
    for field in ("label", "policy_temperature"):
        control_candidate.pop(field, None)
        treatment_candidate.pop(field, None)
    if control_candidate != treatment_candidate:
        raise ValueError(
            "paired campaign candidate inputs differ beyond label and temperature"
        )

    control_runtime = _required_string(
        control.get("runtime_fingerprint"), "control runtime fingerprint"
    )
    treatment_runtime = _required_string(
        treatment.get("runtime_fingerprint"), "treatment runtime fingerprint"
    )
    if control_runtime == treatment_runtime:
        raise ValueError("temperature arms must have distinct runtime fingerprints")
    return {
        "schedule_seed": seed,
        "native_library_sha256": native_sha256,
        "belief_fingerprint": control_belief,
        "control_runtime_fingerprint": control_runtime,
        "treatment_runtime_fingerprint": treatment_runtime,
        "execution_config_identical": True,
        "maximum_engine_steps": maximum_engine_steps,
    }


def participant(summary: dict[str, Any], role: str) -> dict[str, Any]:
    """Return one required participant mapping."""
    value = summary.get(role)
    if not isinstance(value, dict):
        raise ValueError(f"summary {role} participant is missing")
    return cast(dict[str, Any], value)


def _validate_manifest_participant(
    manifest: dict[str, Any], summary: dict[str, Any], *, label: str
) -> None:
    for field in (*_IDENTITY_FIELDS, "label", "policy_temperature"):
        if manifest.get(field) != summary.get(field):
            raise ValueError(f"{label} field {field} differs from summary")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing or invalid")
    return cast(dict[str, Any], value)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing or invalid")
    return value


def _temperature(value: dict[str, Any], label: str) -> float:
    raw = value.get("policy_temperature")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} policy temperature is invalid")
    result = float(raw)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} policy temperature is invalid")
    return result


def _require_expected_temperature(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(expected) or expected < 0.0:
        raise ValueError(f"expected {label} policy temperature is invalid")
    if not math.isclose(actual, expected, abs_tol=1e-12):
        raise ValueError(f"{label} policy temperature differs from expected")


def campaign_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Recompute the native-gauntlet identity from a manifest payload."""
    payload = dict(manifest)
    payload.pop("campaign_fingerprint", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"ptcg-rl/native-checkpoint-gauntlet/v1\0" + encoded
    ).hexdigest()


__all__ = [
    "campaign_manifest_fingerprint",
    "participant",
    "read_manifest",
    "read_summary",
    "validate_manifests",
    "validate_summaries",
]
