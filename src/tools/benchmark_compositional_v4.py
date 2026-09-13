"""Benchmark routed DCCR-v4 inference and grouped exact-residual execution."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model.compositional_projection import SharedCompositionalLinear
from ptcg_rl.model.deck_conditioning import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DeckRoutePlan,
    resolve_deck_route_plan,
)
from ptcg_rl.model.network import AgentPolicyValueNet, build_agent_policy_value_net
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE, StateBatch
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.compositional_transition_pair_io import (
    checkpoint_model_config,
    checkpoint_state_dict,
)
from ptcg_rl.rl.training import (
    RLTrainConfig,
    _load_hydra_config,
    _resolved_target_model_config,
)
from ptcg_rl.training.host_policy import require_cuda_training_host


def main() -> int:
    """Load one converted checkpoint and write reproducible H200 measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--token-count", type=int, default=112)
    parser.add_argument("--option-count", type=int, default=56)
    parser.add_argument("--attachment-count", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    _validate_benchmark_sizes(args)
    require_cuda_training_host()

    raw = _load_hydra_config(args.config_name)
    config = RLTrainConfig.model_validate(raw)
    config = config.model_copy(update={"model": _resolved_target_model_config(config)})
    conditioning = config.model.deck_conditioning
    if (
        conditioning is None
        or conditioning.architecture_version
        != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        or conditioning.compositional is None
        or conditioning.compositional.export_mode != "routed"
    ):
        raise ValueError("benchmark profile must resolve to routed DCCR-v4")

    checkpoint_path = deck_records.repo_path(args.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_model_config(payload) != config.model:
        raise ValueError("benchmark checkpoint model config differs from profile")
    model = build_agent_policy_value_net(config.model).cpu()
    model.load_state_dict(checkpoint_state_dict(payload), strict=True)
    del payload
    gc.collect()
    device = torch.device("cuda")
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    homogeneous_decks = _benchmark_decks(
        conditioning.expert_routes,
        batch_size=args.batch_size,
        mixed=False,
        device=device,
    )
    mixed_decks = _benchmark_decks(
        conditioning.expert_routes,
        batch_size=args.batch_size,
        mixed=True,
        device=device,
    )
    states = _state_batch(
        batch_size=args.batch_size,
        token_count=args.token_count,
        attachment_count=args.attachment_count,
        device=device,
    )
    options = _option_batch(
        batch_size=args.batch_size,
        option_count=args.option_count,
        device=device,
    )

    def homogeneous_forward() -> Tensor:
        return _model_forward(model, states, options, homogeneous_decks)

    def mixed_forward() -> Tensor:
        return _model_forward(model, states, options, mixed_decks)

    homogeneous_metrics = _cuda_benchmark(
        homogeneous_forward,
        warmup=args.warmup,
        samples=args.samples,
        rows=args.batch_size,
    )
    mixed_metrics = _cuda_benchmark(
        mixed_forward,
        warmup=args.warmup,
        samples=args.samples,
        rows=args.batch_size,
    )

    projection = max(
        (
            module
            for module in model.modules()
            if isinstance(module, SharedCompositionalLinear)
        ),
        key=lambda module: module.in_features * module.out_features,
    )
    projection_metrics = _projection_benchmark(
        model,
        projection,
        decks=mixed_decks,
        token_count=args.token_count,
        warmup=args.warmup,
        samples=args.samples,
    )
    named_parameters = tuple(model.named_parameters())
    private_parameters = sum(
        parameter.numel()
        for name, parameter in named_parameters
        if name.startswith("exact_capsules.")
    )
    parameter_count = sum(parameter.numel() for _name, parameter in named_parameters)
    report = {
        "schema": "dccr-v4-h200-throughput-benchmark-v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_name": args.config_name,
        "checkpoint": deck_records.display_path(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "device": torch.cuda.get_device_name(device),
        "geometry": {
            "batch_size": args.batch_size,
            "token_count": args.token_count,
            "option_count": args.option_count,
            "attachment_count": args.attachment_count,
            "route_count": len(conditioning.expert_routes),
            "warmup": args.warmup,
            "samples": args.samples,
        },
        "parameters": {
            "total": parameter_count,
            "exact_capsules": private_parameters,
            "shared": parameter_count - private_parameters,
            "fp32_bytes": parameter_count * 4,
        },
        "actor_forward": {
            "homogeneous_route": homogeneous_metrics,
            "mixed_routes": mixed_metrics,
        },
        "largest_projection": projection_metrics,
        "peak_cuda_memory_bytes_after_model_load": (
            torch.cuda.max_memory_allocated(device)
        ),
    }
    output_path = deck_records.repo_path(args.output).resolve()
    atomic_write_bytes(output_path, json_payload(report), overwrite=True)
    print(json_payload(report).decode("utf-8"), end="")
    return 0


def _validate_benchmark_sizes(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "token_count",
        "option_count",
        "warmup",
        "samples",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.attachment_count < 0:
        raise ValueError("attachment_count must be non-negative")


def _benchmark_decks(
    routes: tuple[Any, ...],
    *,
    batch_size: int,
    mixed: bool,
    device: torch.device,
) -> DeckBatch:
    if not routes:
        raise ValueError("benchmark requires at least one exact route")
    cards = tuple(
        routes[index % len(routes) if mixed else 0].canonical_card_ids
        for index in range(batch_size)
    )
    return DeckBatch.from_card_ids(cards, device=device)


def _state_batch(
    *,
    batch_size: int,
    token_count: int,
    attachment_count: int,
    device: torch.device,
) -> StateBatch:
    shape = (batch_size, token_count)
    attachment_shape = (batch_size, attachment_count)
    return StateBatch(
        card_ids=torch.zeros(shape, dtype=torch.long, device=device),
        areas=torch.zeros(shape, dtype=torch.long, device=device),
        owner_roles=torch.zeros(shape, dtype=torch.long, device=device),
        token_kinds=torch.zeros(shape, dtype=torch.long, device=device),
        scalars=torch.zeros((*shape, TOKEN_SCALAR_SIZE), device=device),
        last_attack_ids=torch.zeros(shape, dtype=torch.long, device=device),
        padding_mask=torch.zeros(shape, dtype=torch.bool, device=device),
        attachment_card_ids=torch.zeros(
            attachment_shape,
            dtype=torch.long,
            device=device,
        ),
        attachment_parent_indices=torch.zeros(
            attachment_shape,
            dtype=torch.long,
            device=device,
        ),
        attachment_kinds=torch.zeros(
            attachment_shape,
            dtype=torch.long,
            device=device,
        ),
    )


def _option_batch(
    *,
    batch_size: int,
    option_count: int,
    device: torch.device,
) -> OptionBatch:
    shape = (batch_size, option_count)
    return OptionBatch(
        option_types=torch.zeros(shape, dtype=torch.long, device=device),
        contexts=torch.zeros(shape, dtype=torch.long, device=device),
        entity_slots=torch.zeros((*shape, 2), dtype=torch.long, device=device),
        entity_slot_mask=torch.zeros(
            (*shape, 2),
            dtype=torch.bool,
            device=device,
        ),
        attack_ids=torch.zeros(shape, dtype=torch.long, device=device),
        card_ids=torch.zeros(shape, dtype=torch.long, device=device),
        scalars=torch.zeros((*shape, SCALAR_FEATURE_SIZE), device=device),
        dynamic_effect_features=torch.zeros(
            (*shape, DYNAMIC_EFFECT_FEATURE_SIZE),
            device=device,
        ),
        dynamic_effect_masks=torch.zeros(
            shape,
            dtype=torch.bool,
            device=device,
        ),
        valid_options=torch.ones(shape, dtype=torch.bool, device=device),
        min_counts=torch.zeros((batch_size,), dtype=torch.long, device=device),
        max_counts=torch.ones((batch_size,), dtype=torch.long, device=device),
    )


def _model_forward(
    model: AgentPolicyValueNet,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
) -> Tensor:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(states, options, decks)
        return cast(Tensor, output.policy_logits)


def _projection_benchmark(
    model: AgentPolicyValueNet,
    projection: SharedCompositionalLinear,
    *,
    decks: DeckBatch,
    token_count: int,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    conditioning = cast(Any, model.config.deck_conditioning)
    grouped_plan = resolve_deck_route_plan(decks, conditioning)
    deck_embeddings = torch.randn(
        (len(decks), projection.deck_dim),
        device=decks.card_ids.device,
    )
    grouped_plan = replace(grouped_plan, deck_embeddings=deck_embeddings)
    fallback_plan: DeckRoutePlan = replace(grouped_plan, lora_dispatch=None)
    base = nn.Linear(
        projection.in_features,
        projection.out_features,
        bias=True,
        device=decks.card_ids.device,
    ).eval()
    inputs = torch.randn(
        (len(decks), token_count, projection.in_features),
        device=decks.card_ids.device,
    )

    def forward(plan: DeckRoutePlan) -> Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return cast(
                Tensor,
                projection(
                    base,
                    inputs,
                    route_plan=plan,
                    capsules=model.exact_capsules,
                ),
            )

    grouped_output = forward(grouped_plan)
    fallback_output = forward(fallback_plan)
    parity = float((grouped_output - fallback_output).abs().max().float().item())
    grouped = _cuda_benchmark(
        lambda: forward(grouped_plan),
        warmup=warmup,
        samples=samples,
        rows=len(decks) * token_count,
    )
    fallback = _cuda_benchmark(
        lambda: forward(fallback_plan),
        warmup=warmup,
        samples=samples,
        rows=len(decks) * token_count,
    )
    return {
        "target": projection.target,
        "domain": projection.domain,
        "in_features": projection.in_features,
        "out_features": projection.out_features,
        "shared_rank_total": projection.basis_count * projection.shared_rank,
        "grouped": grouped,
        "fallback": fallback,
        "grouped_speedup": fallback["mean_ms"] / grouped["mean_ms"],
        "max_abs_parity_error": parity,
    }


def _cuda_benchmark(
    operation: Callable[[], Tensor],
    *,
    warmup: int,
    samples: int,
    rows: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _ = operation()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        _ = operation()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    mean_ms = statistics.fmean(timings)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings),
        "p95_ms": ordered[p95_index],
        "rows_per_second": rows * 1_000.0 / mean_ms,
    }


if __name__ == "__main__":
    raise SystemExit(main())
