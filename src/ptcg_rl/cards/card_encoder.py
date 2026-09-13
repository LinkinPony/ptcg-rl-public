"""Residual card-ID and static-feature encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor, nn

from ptcg_rl.activation_precision import preserve_cuda_bfloat16_activation
from ptcg_rl.cards.static_features import (
    DEFAULT_NUM_CARD_IDS,
    FEATURE_SIZE,
    StaticFeatureMmapMode,
    load_static_feature_table,
)


class CardEncoderConfig(BaseModel):
    """Config for the residual card representation module."""

    model_config = ConfigDict(extra="forbid")

    d_model: int = 128
    hidden_dim: int | None = None
    dropout: float = 0.0
    embedding_l2: float = 1.0e-5
    feature_table_path: Path = Path(
        "outputs/cards/static_features/card_static_features.npy"
    )
    feature_table_mmap_mode: StaticFeatureMmapMode | None = "c"

    @field_validator("d_model")
    @classmethod
    def valid_d_model(cls, value: int) -> int:
        """Reject non-positive model dimensions."""
        if value <= 0:
            raise ValueError("d_model must be positive")
        return value

    @field_validator("hidden_dim")
    @classmethod
    def valid_hidden_dim(cls, value: int | None) -> int | None:
        """Reject non-positive hidden dimensions."""
        if value is not None and value <= 0:
            raise ValueError("hidden_dim must be positive when set")
        return value

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, value: float) -> float:
        """Reject invalid dropout rates."""
        if value < 0.0 or value >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return value

    @field_validator("embedding_l2")
    @classmethod
    def valid_embedding_l2(cls, value: float) -> float:
        """Reject negative L2 coefficients."""
        if value < 0.0:
            raise ValueError("embedding_l2 must be non-negative")
        return value


class CardEncoder(nn.Module):
    """Encode card IDs as ``Embedding(id) + MLP(static_feat[id])``."""

    static_features: Tensor
    static_projection: nn.Sequential

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_card_ids: int = DEFAULT_NUM_CARD_IDS,
        static_feature_size: int = FEATURE_SIZE,
        static_features: npt.NDArray[np.float32] | Tensor | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        embedding_l2: float = 1.0e-5,
    ) -> None:
        """Initialize the residual card encoder."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_card_ids <= 0:
            raise ValueError("num_card_ids must be positive")
        if static_feature_size <= 0:
            raise ValueError("static_feature_size must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if embedding_l2 < 0.0:
            raise ValueError("embedding_l2 must be non-negative")

        self.d_model = d_model
        self.num_card_ids = num_card_ids
        self.static_feature_size = static_feature_size
        self.embedding_l2 = embedding_l2

        static_tensor = _static_feature_tensor(
            static_features,
            num_card_ids=num_card_ids,
            static_feature_size=static_feature_size,
        )
        self.register_buffer("static_features", static_tensor)

        self.embedding = nn.Embedding(
            num_card_ids + 1,
            d_model,
            padding_idx=0,
        )
        width = hidden_dim or d_model
        self.static_projection = nn.Sequential(
            nn.Linear(static_feature_size, width, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, d_model, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize ID residuals near zero while keeping pad exactly zero."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, card_ids: Tensor) -> Tensor:
        """Return encoded card vectors for an arbitrary-shaped ID tensor."""
        indices = self.safe_card_id_indices(card_ids)
        static_rows = self.static_features[indices]
        projected = self.static_projection(static_rows)
        embedded = preserve_cuda_bfloat16_activation(self.embedding(indices))
        return preserve_cuda_bfloat16_activation(embedded + projected)

    def safe_card_id_indices(self, card_ids: Tensor) -> Tensor:
        """Map invalid or unknown card IDs to pad/OOV index 0."""
        ids = card_ids.to(dtype=torch.long)
        return torch.where(
            (ids >= 1) & (ids <= self.num_card_ids),
            ids,
            torch.zeros_like(ids),
        )

    def embedding_l2_loss(self) -> Tensor:
        """Return the configured L2 penalty for non-pad embedding rows."""
        if self.embedding_l2 == 0.0:
            return self.embedding.weight.new_zeros(())
        return self.embedding.weight[1:].pow(2).mean() * self.embedding_l2

    @classmethod
    def from_feature_table(
        cls,
        static_features: npt.NDArray[np.float32] | Tensor,
        *,
        d_model: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        embedding_l2: float = 1.0e-5,
    ) -> CardEncoder:
        """Create an encoder using an already-loaded feature table."""
        shape = tuple(static_features.shape)
        if len(shape) != 2:
            raise ValueError("static_features must be rank-2")
        return cls(
            d_model=d_model,
            num_card_ids=shape[0] - 1,
            static_feature_size=shape[1],
            static_features=static_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            embedding_l2=embedding_l2,
        )

    @classmethod
    def from_feature_table_path(
        cls,
        path: Path,
        *,
        d_model: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        embedding_l2: float = 1.0e-5,
        mmap_mode: StaticFeatureMmapMode | None = "c",
    ) -> CardEncoder:
        """Load static features from NPZ/NPY and create an encoder."""
        return cls.from_feature_table(
            load_static_feature_table(
                _resolve_feature_table_path(path),
                mmap_mode=mmap_mode,
            ),
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            embedding_l2=embedding_l2,
        )

    @classmethod
    def from_npz(
        cls,
        path: Path,
        *,
        d_model: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        embedding_l2: float = 1.0e-5,
        mmap_mode: StaticFeatureMmapMode | None = "c",
    ) -> CardEncoder:
        """Load static features from a legacy NPZ path or newer NPY path."""
        return cls.from_feature_table_path(
            path,
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            embedding_l2=embedding_l2,
            mmap_mode=mmap_mode,
        )


def build_card_encoder(config: CardEncoderConfig) -> CardEncoder:
    """Build ``CardEncoder`` from a validated config."""
    return CardEncoder.from_feature_table_path(
        config.feature_table_path,
        d_model=config.d_model,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        embedding_l2=config.embedding_l2,
        mmap_mode=config.feature_table_mmap_mode,
    )


def _resolve_feature_table_path(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    module_path = Path(__file__).resolve()
    for root in (Path.cwd(), module_path.parents[2], module_path.parents[3]):
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


def _static_feature_tensor(
    static_features: npt.NDArray[np.float32] | Tensor | None,
    *,
    num_card_ids: int,
    static_feature_size: int,
) -> Tensor:
    if static_features is None:
        return torch.zeros(
            (num_card_ids + 1, static_feature_size),
            dtype=torch.float32,
        )
    if isinstance(static_features, np.ndarray):
        tensor = torch.from_numpy(np.asarray(static_features, dtype=np.float32))
    else:
        tensor = static_features.detach().to(dtype=torch.float32)
    expected_shape = (num_card_ids + 1, static_feature_size)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"static_features must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("static_features contains NaN or infinite values")
    if not torch.all(tensor[0] == 0.0):
        raise ValueError("static_features row 0 must be all zeros")
    return tensor
