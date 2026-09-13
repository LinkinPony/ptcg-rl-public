"""Central activation-precision boundaries for mixed-precision execution."""

from __future__ import annotations

import torch
from torch import Tensor


def preserve_cuda_bfloat16_activation(tensor: Tensor) -> Tensor:
    """Keep a long-lived activation in BF16 during CUDA BF16 autocast.

    Learners retain FP32 master parameters, so embeddings, direct parameters,
    and LayerNorm can produce FP32 tensors inside autocast.  Casting only at
    explicit activation boundaries prevents those tensors from promoting an
    entire residual stream while preserving explicit FP32 inference and every
    CPU execution mode.
    """
    if (
        tensor.device.type == "cuda"
        and tensor.dtype == torch.float32
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    ):
        return tensor.to(dtype=torch.bfloat16)
    return tensor


__all__ = ["preserve_cuda_bfloat16_activation"]
