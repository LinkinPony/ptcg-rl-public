"""Device-aware scalar tensor assertions without CUDA host synchronization."""

from __future__ import annotations

import torch
from torch import Tensor


def require_tensor_condition(condition: Tensor, message: str) -> None:
    """Raise for a false scalar while keeping CUDA validation asynchronous."""
    if condition.numel() != 1 or condition.dtype != torch.bool:
        raise TypeError("tensor validation requires one boolean scalar")
    if condition.is_meta:
        return
    if condition.is_cuda:
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise ValueError(message)


__all__ = ["require_tensor_condition"]
