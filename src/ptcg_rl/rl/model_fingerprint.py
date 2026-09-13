"""Canonical content identity for a fully constructed serving model state."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from ptcg_rl.model.network import (
    SCHEMA9_WARM_START_INITIALIZATION_SEED,
    AgentNetworkConfig,
    build_agent_policy_value_net,
)
from ptcg_rl.model.weights_only_migration_classification import (
    checkpoint_model_config_payload,
    checkpoint_publish_version,
    checkpoint_state_dict,
)

_MODEL_STATE_DOMAIN = b"ptcg-rl/full-model-state/v1\x00"


def canonical_model_state_fingerprint(
    model_or_state: nn.Module | Mapping[str, Tensor],
) -> str:
    """Hash every named tensor, including migrated parameters and buffers."""
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    if not state:
        raise ValueError("model state fingerprint requires at least one tensor")
    digest = hashlib.sha256()
    digest.update(_MODEL_STATE_DOMAIN)
    digest.update(struct.pack(">Q", len(state)))
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"model state value is not a tensor: {name}")
        encoded_name = str(name).encode("utf-8")
        encoded_dtype = str(tensor.dtype).removeprefix("torch.").encode("ascii")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        byte_array = (
            tensor.detach()
            .contiguous()
            .view(torch.uint8)
            .to(device="cpu")
            .numpy()
        )
        # ``hashlib`` accepts the contiguous buffer directly.  Avoiding
        # ``ndarray.tobytes()`` removes another potentially model-sized host
        # allocation from every learner publication.
        payload = byte_array.data.cast("B")
        digest.update(struct.pack(">Q", payload.nbytes))
        digest.update(payload)
    return digest.hexdigest()


def constructed_checkpoint_fingerprint(
    checkpoint_path: Path | str,
    *,
    migration_seed: int = SCHEMA9_WARM_START_INITIALIZATION_SEED,
) -> tuple[str, int | None]:
    """Load one checkpoint through the production migration and hash full state."""
    if migration_seed != SCHEMA9_WARM_START_INITIALIZATION_SEED:
        raise ValueError("schema-9 migration seed is a fixed compatibility contract")
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    config = AgentNetworkConfig.model_validate(
        checkpoint_model_config_payload(checkpoint)
    )
    model = build_agent_policy_value_net(config)
    incompatible = model.load_state_dict(checkpoint_state_dict(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint differs from the constructed serving architecture: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return (
        canonical_model_state_fingerprint(model),
        checkpoint_publish_version(checkpoint),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--migration-seed",
        type=int,
        default=SCHEMA9_WARM_START_INITIALIZATION_SEED,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print a machine-readable full-state identity for profile preregistration."""
    args = _parser().parse_args(argv)
    fingerprint, policy_version = constructed_checkpoint_fingerprint(
        args.checkpoint,
        migration_seed=args.migration_seed,
    )
    print(
        json.dumps(
            {
                "model_fingerprint": fingerprint,
                "policy_version": policy_version,
                "migration_seed": SCHEMA9_WARM_START_INITIALIZATION_SEED,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "canonical_model_state_fingerprint",
    "constructed_checkpoint_fingerprint",
]
