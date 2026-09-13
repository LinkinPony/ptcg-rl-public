"""Hydra-backed CLI for the persistent component TrueSkill league."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.continuous_league.coordinator import run_coordinator
from ptcg_rl.evaluation.continuous_league.models import (
    ContinuousLeagueConfig,
    ManualCheckpointRequest,
    ManualDeckRequest,
    ManualReleaseRequest,
)
from ptcg_rl.evaluation.continuous_league.native_match import (
    native_match_contract_fingerprints,
    run_native_match_server,
)
from ptcg_rl.evaluation.continuous_league.parity import (
    run_checkpoint_action_parity,
)
from ptcg_rl.evaluation.continuous_league.reporting import LeagueRepository
from ptcg_rl.evaluation.continuous_league.worker import LeagueWorker
from ptcg_rl.evaluation.continuous_league.worker_pool import run_worker_pool
from ptcg_rl.training.source_identity import source_manifest_environment_name
from ptcg_rl.training.source_snapshot import (
    materialize_training_source,
    require_clean_main_checkout,
)


def main() -> None:
    """Run one coordinator, worker, inspection, replay, or manual admission action."""
    arguments = _arguments()
    repo_root = records.repo_path(Path(".")).resolve()
    if arguments.command == "coordinator" or (
        arguments.command == "worker" and not arguments.once
    ):
        _require_tmux(arguments.command)
        _enter_immutable_source(repo_root)
    config = _load_config(
        repo_root,
        profile=arguments.profile,
        overrides=tuple(arguments.override),
    )
    if arguments.command == "coordinator":
        run_coordinator(
            config,
            repo_root=repo_root,
            allow_source_revision=arguments.adopt_source_revision,
        )
        return
    if arguments.command == "worker":
        if arguments.concurrency > 1:
            run_worker_pool(
                config,
                worker_id_prefix=arguments.worker_id,
                concurrency=arguments.concurrency,
                repo_root=repo_root,
                coordinator_url=arguments.coordinator_url,
                gpu_index=arguments.gpu_index,
                executor_processes=arguments.executor_processes,
                ignore_resource_load=arguments.ignore_resource_load,
                once=arguments.once,
            )
            return
        LeagueWorker(
            config,
            worker_id=arguments.worker_id,
            repo_root=repo_root,
            coordinator_url=arguments.coordinator_url,
            gpu_index=arguments.gpu_index,
            ignore_resource_load=arguments.ignore_resource_load,
        ).run(once=arguments.once)
        return
    if arguments.command == "native-match":
        run_native_match_server(
            config,
            repo_root=repo_root,
            concurrency=arguments.concurrency,
        )
        return
    if arguments.command == "parity-check":
        payload = run_checkpoint_action_parity(
            config,
            repo_root=repo_root,
            checkpoint_pair_path=arguments.checkpoint_pair,
            deck_path=arguments.deck,
            opponent_script=arguments.opponent_script,
            candidate_seat=arguments.candidate_seat,
            output_path=arguments.output,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if arguments.command == "status":
        database_path = (
            config.database_path
            if config.database_path.is_absolute()
            else repo_root / config.database_path
        )
        repository = LeagueRepository(database_path)
        payload = {
            "summary": repository.summary().model_dump(mode="json"),
            "checkpoints": [
                item.model_dump(mode="json")
                for item in repository.standings("controllers")
                if item.kind == "checkpoint"
            ],
            "decks": [
                item.model_dump(mode="json") for item in repository.standings("decks")
            ],
            "bundles": [
                item.model_dump(mode="json") for item in repository.standings("bundles")
            ],
            "workers": [item.model_dump(mode="json") for item in repository.workers()],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    base_url = (
        arguments.coordinator_url
        or f"http://{config.coordinator_host}:{config.coordinator_port}"
    ).rstrip("/")
    if arguments.command == "rebuild-ratings":
        payload = _post(base_url, "/control/v1/rebuild-ratings?confirm=true", {})
    elif arguments.command == "add-checkpoint":
        payload = _post(
            base_url,
            "/control/v1/add-checkpoint",
            ManualCheckpointRequest(
                pair_manifest_path=arguments.path,
                label=arguments.label,
            ).model_dump(mode="json"),
        )
    elif arguments.command == "add-deck":
        payload = _post(
            base_url,
            "/control/v1/add-deck",
            ManualDeckRequest(
                deck_path=arguments.path,
                label=arguments.label,
            ).model_dump(mode="json"),
        )
    else:
        payload = _post(
            base_url,
            "/control/v1/add-release",
            ManualReleaseRequest(
                release_manifest_path=arguments.path,
                alias=arguments.alias,
            ).model_dump(mode="json"),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--coordinator-url")
    subparsers = parser.add_subparsers(dest="command", required=True)
    coordinator = subparsers.add_parser("coordinator")
    coordinator.add_argument("--adopt-source-revision", action="store_true")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--concurrency", type=int, default=1)
    worker.add_argument("--gpu-index", type=int)
    worker.add_argument("--executor-processes", type=int, default=1)
    worker.add_argument("--ignore-resource-load", action="store_true")
    native_match = subparsers.add_parser("native-match")
    native_match.add_argument("--concurrency", type=int, default=1)
    parity = subparsers.add_parser("parity-check")
    parity.add_argument("--checkpoint-pair", type=Path, required=True)
    parity.add_argument("--deck", type=Path, required=True)
    parity.add_argument("--opponent-script", default="end")
    parity.add_argument("--candidate-seat", type=int, choices=(0, 1), default=0)
    parity.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("rebuild-ratings")
    checkpoint = subparsers.add_parser("add-checkpoint")
    checkpoint.add_argument("path", type=Path)
    checkpoint.add_argument("--label")
    deck = subparsers.add_parser("add-deck")
    deck.add_argument("path", type=Path)
    deck.add_argument("--label")
    release = subparsers.add_parser("add-release")
    release.add_argument("path", type=Path)
    release.add_argument("--alias", required=True)
    arguments = parser.parse_args()
    if arguments.command == "worker":
        if arguments.concurrency <= 0:
            parser.error("--concurrency must be positive")
        if arguments.gpu_index is not None and arguments.gpu_index < 0:
            parser.error("--gpu-index must be non-negative")
        if arguments.executor_processes <= 0:
            parser.error("--executor-processes must be positive")
        if arguments.concurrency > 1 and (
            arguments.concurrency % arguments.executor_processes
            or arguments.concurrency // arguments.executor_processes <= 1
        ):
            parser.error(
                "--concurrency must divide into at least two lanes per executor process"
            )
    if arguments.command == "native-match" and arguments.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return arguments


def _load_config(
    repo_root: Path,
    *,
    profile: str,
    overrides: tuple[str, ...],
) -> ContinuousLeagueConfig:
    config_dir = repo_root / "configs" / "evaluation" / "continuous_league"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        hydra_config = compose(config_name=profile, overrides=list(overrides))
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("continuous league Hydra config must resolve to a mapping")
    config = ContinuousLeagueConfig.model_validate(cast(dict[str, Any], raw))
    if config.native_match_command:
        runtime_fingerprint, belief_fingerprint = native_match_contract_fingerprints(
            config.native_match
        )
        if (
            runtime_fingerprint != config.runtime_fingerprint
            or belief_fingerprint != config.belief_fingerprint
        ):
            raise ValueError(
                "native match profile has stale runtime/belief fingerprints"
            )
    return config


def _post(base_url: str, path: str, payload: object) -> Any:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600.0) as response:
        return json.loads(response.read())


def _require_tmux(role: str) -> None:
    if not os.environ.get("TMUX"):
        raise RuntimeError(f"formal continuous league {role} must start inside tmux")


def _enter_immutable_source(repo_root: Path) -> None:
    """Re-exec a formal daemon from the clean main commit's no-.git snapshot."""
    manifest_name = source_manifest_environment_name()
    if os.environ.get(manifest_name):
        return
    require_clean_main_checkout(repo_root)
    source = materialize_training_source(repo_root, revision="HEAD")
    environment = dict(os.environ)
    environment[manifest_name] = str(source.manifest_path)
    python_paths = (
        str(source.root / "data" / "sample_submission"),
        str(source.root / "src"),
    )
    existing = tuple(
        item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item
    )
    environment["PYTHONPATH"] = os.pathsep.join((*python_paths, *existing))
    command = (
        sys.executable,
        str(source.root / "src" / "tools" / "continuous_league.py"),
        *sys.argv[1:],
    )
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
