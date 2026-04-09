"""
Shared input validation utilities for the imagen package.

Requirements: 1.1, 4.2
"""

import torch
from typing import Tuple


def validate_input_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple,
    name: str = "input",
) -> None:
    """Raise ValueError if tensor shape does not match expected_shape.

    Dimensions set to -1 in expected_shape are treated as wildcards (any size).

    Args:
        tensor: The tensor to validate.
        expected_shape: Tuple of expected sizes; use -1 for "any".
        name: Human-readable name used in error messages.

    Raises:
        TypeError: If tensor is not a torch.Tensor.
        ValueError: If the number of dimensions or any fixed dimension mismatches.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
        )

    if tensor.dim() != len(expected_shape):
        raise ValueError(
            f"{name} must be {len(expected_shape)}D, "
            f"got {tensor.dim()}D tensor with shape {tuple(tensor.shape)}"
        )

    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} dimension {i}: expected {expected}, got {actual}. "
                f"Full shape: expected {expected_shape}, got {tuple(tensor.shape)}"
            )


def validate_finite(tensor: torch.Tensor, name: str = "input") -> None:
    """Raise ValueError if tensor contains NaN or Inf values.

    Args:
        tensor: The tensor to check.
        name: Human-readable name used in error messages.

    Raises:
        ValueError: If any NaN or Inf values are present.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
