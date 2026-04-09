"""ConditioningMLP — placeholder, implemented in task 2."""

import torch.nn as nn


class ConditioningMLP(nn.Module):
    """Projects a conditioning vector to a spatial bias for the U-Net bottleneck."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        raise NotImplementedError("ConditioningMLP is implemented in task 2")
