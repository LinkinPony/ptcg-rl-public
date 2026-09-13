"""Weighted dense-expert factorization for DCCR architecture transitions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TruncatedSvdConfig:
    """Deterministic randomized-SVD controls for one topology conversion."""

    seed: int = 0
    oversampling: int = 8
    power_iterations: int = 2
    algorithm: str = "deterministic-randomized-svd-v1"

    def __post_init__(self) -> None:
        """Reject settings that cannot define a stable bounded factorization."""
        if self.seed < 0:
            raise ValueError("randomized SVD seed must be non-negative")
        if self.oversampling < 0:
            raise ValueError("randomized SVD oversampling must be non-negative")
        if self.power_iterations < 0:
            raise ValueError("randomized SVD power iterations must be non-negative")
        if self.algorithm != "deterministic-randomized-svd-v1":
            raise ValueError("unsupported truncated SVD algorithm")

    def manifest(self) -> dict[str, int | str]:
        """Return the immutable fitting recipe recorded by transition assets."""
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "oversampling": self.oversampling,
            "power_iterations": self.power_iterations,
        }


@dataclass(frozen=True)
class CompositionalFactorization:
    """One projection's shared mean, basis bank, and exact remainders."""

    mean: Tensor
    shared_a: Tensor
    shared_b: Tensor
    coefficients: Tensor
    exact_a: Tensor
    exact_b: Tensor
    singular_values: Tensor
    weighted_relative_error: float
    svd_config: TruncatedSvdConfig

    def reconstructed_weights(self) -> Tensor:
        """Materialize every expert weight represented by this factorization."""
        bases = torch.matmul(self.shared_b, self.shared_a)
        shared = torch.einsum("dk,koi->doi", self.coefficients, bases)
        exact = torch.matmul(self.exact_b, self.exact_a)
        return self.mean.unsqueeze(0) + shared + exact


@dataclass(frozen=True)
class RouterFit:
    """Closed-form last-layer router fit and resulting exact route biases."""

    raw_targets: Tensor
    predictions: Tensor
    route_biases: Tensor
    root_mean_squared_error: float
    leave_one_out_root_mean_squared_error: float


def factorize_dense_expert_weights(
    expert_weights: Tensor,
    *,
    sampling_weights: Tensor,
    basis_count: int,
    shared_rank: int,
    exact_rank: int,
    coefficient_limit: float = 1.95,
    ridge: float = 1.0e-6,
    svd_config: TruncatedSvdConfig | None = None,
) -> CompositionalFactorization:
    """Factor ``[D, out, in]`` experts into shared and exact low-rank terms."""
    if expert_weights.ndim != 3 or int(expert_weights.shape[0]) <= 0:
        raise ValueError("expert weights must have shape [D, out, in]")
    if not expert_weights.is_floating_point():
        raise TypeError("expert weights must be floating point")
    if min(basis_count, shared_rank, exact_rank) <= 0:
        raise ValueError("factorization ranks and basis count must be positive")
    if not math.isfinite(coefficient_limit) or not 0.0 < coefficient_limit < 2.0:
        raise ValueError("coefficient_limit must be finite and in (0, 2)")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    fit_config = svd_config or TruncatedSvdConfig()
    deck_count = int(expert_weights.shape[0])
    if tuple(sampling_weights.shape) != (deck_count,):
        raise ValueError("sampling weights must have shape [D]")
    if not bool(torch.isfinite(sampling_weights).all().item()) or bool(
        (sampling_weights <= 0.0).any().item()
    ):
        raise ValueError("sampling weights must be finite and positive")

    original_device = expert_weights.device
    values = expert_weights.float()
    probabilities = sampling_weights.to(device=values.device, dtype=torch.float32)
    probabilities = probabilities / probabilities.sum()
    mean = torch.einsum("d,doi->oi", probabilities, values)
    deltas = values - mean.unsqueeze(0)
    flattened = deltas.reshape(deck_count, -1)
    weighted = flattened * probabilities.sqrt().unsqueeze(1)
    task_gram = weighted @ weighted.transpose(0, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(task_gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    eigenvectors = eigenvectors.index_select(1, order)
    singular_values = eigenvalues.sqrt()

    out_features, in_features = values.shape[1:]
    shared_a_rows: list[Tensor] = []
    shared_b_rows: list[Tensor] = []
    positive_tolerance = max(float(eigenvalues.max().item()), 1.0) * 1.0e-10
    for basis_index in range(basis_count):
        if (
            basis_index >= deck_count
            or float(eigenvalues[basis_index].item()) <= positive_tolerance
        ):
            basis_matrix = values.new_zeros((out_features, in_features))
        else:
            task_direction = eigenvectors[:, basis_index]
            basis_flat = task_direction @ weighted
            basis_flat = basis_flat / singular_values[basis_index].clamp_min(1.0e-12)
            basis_matrix = basis_flat.reshape(out_features, in_features)
        factor_a, factor_b = _truncated_svd_factors(
            basis_matrix,
            rank=shared_rank,
            config=fit_config,
            stream=basis_index,
        )
        shared_a_rows.append(factor_a)
        shared_b_rows.append(factor_b)
    shared_a = torch.stack(shared_a_rows)
    shared_b = torch.stack(shared_b_rows)
    basis_matrices = torch.matmul(shared_b, shared_a)

    basis_flat = basis_matrices.reshape(basis_count, -1)
    basis_gram = basis_flat @ basis_flat.transpose(0, 1)
    right_hand_side = flattened @ basis_flat.transpose(0, 1)
    regularized = basis_gram + ridge * torch.eye(
        basis_count,
        device=values.device,
        dtype=values.dtype,
    )
    coefficients = torch.linalg.solve(regularized, right_hand_side.transpose(0, 1))
    coefficients = coefficients.transpose(0, 1)

    max_abs = coefficients.abs().amax(dim=0)
    basis_scales = torch.maximum(
        max_abs / coefficient_limit,
        torch.ones_like(max_abs),
    )
    shared_b = shared_b * basis_scales[:, None, None]
    coefficients = coefficients / basis_scales.unsqueeze(0)
    coefficients = coefficients.clamp(
        min=-coefficient_limit,
        max=coefficient_limit,
    )
    basis_matrices = torch.matmul(shared_b, shared_a)
    shared_reconstruction = torch.einsum(
        "dk,koi->doi",
        coefficients,
        basis_matrices,
    )
    remainders = deltas - shared_reconstruction

    exact_a_rows: list[Tensor] = []
    exact_b_rows: list[Tensor] = []
    for deck_index, remainder in enumerate(remainders):
        factor_a, factor_b = _truncated_svd_factors(
            remainder,
            rank=exact_rank,
            config=fit_config,
            stream=basis_count + deck_index,
        )
        exact_a_rows.append(factor_a)
        exact_b_rows.append(factor_b)
    exact_a = torch.stack(exact_a_rows)
    exact_b = torch.stack(exact_b_rows)
    reconstruction = (
        mean.unsqueeze(0) + shared_reconstruction + torch.matmul(exact_b, exact_a)
    )
    squared_error = (values - reconstruction).square().sum(dim=(1, 2))
    squared_reference = values.square().sum(dim=(1, 2)).clamp_min(1.0e-20)
    relative_error = torch.sqrt(
        (probabilities * squared_error).sum()
        / (probabilities * squared_reference).sum()
    )
    return CompositionalFactorization(
        mean=mean.to(device=original_device),
        shared_a=shared_a.to(device=original_device),
        shared_b=shared_b.to(device=original_device),
        coefficients=coefficients.to(device=original_device),
        exact_a=exact_a.to(device=original_device),
        exact_b=exact_b.to(device=original_device),
        singular_values=singular_values[:basis_count].to(device=original_device),
        weighted_relative_error=float(relative_error.item()),
        svd_config=fit_config,
    )


def fit_compositional_router(
    router_norm: nn.LayerNorm,
    router: nn.Sequential,
    deck_embeddings: Tensor,
    coefficients: Tensor,
    *,
    ridge: float = 1.0e-3,
) -> RouterFit:
    """Fit the router's last Linear and return exact coefficient residuals."""
    if deck_embeddings.ndim != 2 or coefficients.ndim != 2:
        raise ValueError("router fit inputs must be rank-two tensors")
    if int(deck_embeddings.shape[0]) != int(coefficients.shape[0]):
        raise ValueError("router fit rows must align")
    if not bool((coefficients.abs() < 2.0).all().item()):
        raise ValueError("bounded coefficients must lie strictly inside (-2, 2)")
    if len(router) != 3 or not isinstance(router[0], nn.Linear):
        raise TypeError("router must contain Linear, activation, Linear")
    output = router[2]
    if not isinstance(output, nn.Linear):
        raise TypeError("router output must be Linear")
    if output.out_features != int(coefficients.shape[1]):
        raise ValueError("router output width and coefficient count disagree")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("router ridge must be finite and positive")

    module_device = router[0].weight.device
    module_dtype = router[0].weight.dtype
    embeddings = deck_embeddings.to(device=module_device, dtype=module_dtype)
    raw_targets = 2.0 * torch.atanh(
        coefficients.to(device=module_device, dtype=torch.float32) / 2.0
    )
    with torch.no_grad():
        hidden = router[1](router[0](router_norm(embeddings))).float()
        design = torch.cat(
            (
                hidden,
                torch.ones(
                    (hidden.shape[0], 1),
                    device=hidden.device,
                    dtype=hidden.dtype,
                ),
            ),
            dim=1,
        )
        gram = design.transpose(0, 1) @ design
        regularizer = ridge * torch.eye(
            gram.shape[0],
            device=gram.device,
            dtype=gram.dtype,
        )
        regularizer[-1, -1] = 0.0
        solution = torch.linalg.solve(
            gram + regularizer,
            design.transpose(0, 1) @ raw_targets,
        )
        output.weight.copy_(solution[:-1].transpose(0, 1).to(dtype=output.weight.dtype))
        output.bias.copy_(solution[-1].to(dtype=output.bias.dtype))
        predictions = cast(Tensor, router(router_norm(embeddings))).float()
    route_biases = raw_targets - predictions
    error = torch.sqrt((route_biases.square()).mean())
    if int(design.shape[0]) <= 1:
        leave_one_out_error = error
    else:
        influence_solve = torch.linalg.solve(
            gram + regularizer,
            design.transpose(0, 1),
        )
        leverage = (design * influence_solve.transpose(0, 1)).sum(dim=1)
        leave_one_out_residual = route_biases / (1.0 - leverage).clamp_min(
            1.0e-6
        ).unsqueeze(1)
        leave_one_out_error = torch.sqrt(leave_one_out_residual.square().mean())
    return RouterFit(
        raw_targets=raw_targets,
        predictions=predictions,
        route_biases=route_biases,
        root_mean_squared_error=float(error.item()),
        leave_one_out_root_mean_squared_error=float(leave_one_out_error.item()),
    )


def _truncated_svd_factors(
    matrix: Tensor,
    *,
    rank: int,
    config: TruncatedSvdConfig,
    stream: int,
) -> tuple[Tensor, Tensor]:
    """Return padded factors from a deterministic randomized truncated SVD."""
    if matrix.ndim != 2 or rank <= 0:
        raise ValueError("SVD factorization requires a matrix and positive rank")
    if stream < 0:
        raise ValueError("randomized SVD stream must be non-negative")
    available_rank = min(rank, int(matrix.shape[0]), int(matrix.shape[1]))
    if not bool(torch.count_nonzero(matrix).item()):
        return (
            matrix.new_zeros((rank, matrix.shape[1])),
            matrix.new_zeros((matrix.shape[0], rank)),
        )
    minimum_dimension = min(int(matrix.shape[0]), int(matrix.shape[1]))
    sketch_width = min(
        minimum_dimension,
        available_rank + config.oversampling,
    )
    if sketch_width == minimum_dimension:
        left, singular_values, right = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
    else:
        generator = torch.Generator(device=matrix.device)
        generator.manual_seed(_randomized_svd_seed(config, stream=stream))
        omega = torch.randn(
            (int(matrix.shape[1]), sketch_width),
            device=matrix.device,
            dtype=matrix.dtype,
            generator=generator,
        )
        sample = matrix @ omega
        for _ in range(config.power_iterations):
            left_sample = torch.linalg.qr(sample, mode="reduced").Q
            right_sample = torch.linalg.qr(
                matrix.transpose(0, 1) @ left_sample,
                mode="reduced",
            ).Q
            sample = matrix @ right_sample
        basis = torch.linalg.qr(sample, mode="reduced").Q
        reduced = basis.transpose(0, 1) @ matrix
        reduced_left, singular_values, right = torch.linalg.svd(
            reduced,
            full_matrices=False,
        )
        left = basis @ reduced_left
    roots = singular_values[:available_rank].clamp_min(0.0).sqrt()
    factor_a = roots.unsqueeze(1) * right[:available_rank]
    factor_b = left[:, :available_rank] * roots.unsqueeze(0)
    if available_rank < rank:
        factor_a = torch.nn.functional.pad(
            factor_a,
            (0, 0, 0, rank - available_rank),
        )
        factor_b = torch.nn.functional.pad(
            factor_b,
            (0, rank - available_rank),
        )
    return factor_a, factor_b


def _randomized_svd_seed(config: TruncatedSvdConfig, *, stream: int) -> int:
    """Derive independent stable generator streams without global RNG mutation."""
    payload = f"{config.algorithm}\0{config.seed}\0{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


__all__ = [
    "CompositionalFactorization",
    "RouterFit",
    "TruncatedSvdConfig",
    "factorize_dense_expert_weights",
    "fit_compositional_router",
]
