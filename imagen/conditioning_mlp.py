"""
ConditioningMLP — projects a 64-float conditioning vector to a spatial bias
that is broadcast-added to the U-Net bottleneck feature map.

Requirements: 2.1, 2.2, 2.3
"""

import torch
import torch.nn as nn


class ConditioningMLP(nn.Module):
    """Two-layer MLP that projects a conditioning vector to a bottleneck bias.

    Architecture:
        Linear(in_dim → hidden_dim) → ReLU → Linear(hidden_dim → out_dim)

    The output shape is ``(B, out_dim)``, which can be reshaped to
    ``(B, out_dim, 1, 1)`` and broadcast-added to a bottleneck feature map
    of shape ``(B, out_dim, 1, F)``.

    Args:
        in_dim:     Dimensionality of the input conditioning vector (default 64).
        hidden_dim: Width of the hidden layer (default 128).
        out_dim:    Output dimensionality; should match bottleneck channel count.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Project conditioning vector.

        Args:
            cond: Tensor of shape ``(B, in_dim)``.

        Returns:
            Tensor of shape ``(B, out_dim)``.
        """
        x = self.fc1(cond)
        x = self.relu(x)
        x = self.fc2(x)
        return x
