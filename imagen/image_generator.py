"""ImageGenerator — placeholder, implemented in task 3."""

import torch.nn as nn


class ImageGenerator(nn.Module):
    """Asymmetric 2D U-Net that predicts noise for diffusion-style training."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        cond_dim: int = 64,
        cond_hidden: int = 128,
    ):
        super().__init__()
        raise NotImplementedError("ImageGenerator is implemented in task 3")
