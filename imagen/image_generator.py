"""
ImageGenerator — asymmetric 2D U-Net for diffusion-style noise prediction.

Input:  (batch, 2, 11, 836)  — noisy spectrogram window
Output: (batch, 2, 11, 836)  — predicted noise

Requirements: 1.1–1.6, 2.1–2.3, 4.1, 7.3
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .conditioning_mlp import ConditioningMLP
from .utils import validate_input_shape, validate_finite


class ImageGenerator(nn.Module):
    """Asymmetric 2D U-Net that predicts noise for diffusion-style training.

    Architecture summary
    --------------------
    Encoder:
        time_down   : Conv2d(2→64,   k=(11,1), s=(11,1))   (B,2,11,836)→(B,64,1,836)
        freq_down1  : Conv2d(64→128, k=(1,3),  s=(1,2), p=(0,1))  →(B,128,1,418)
        freq_down2  : Conv2d(128→256,k=(1,3),  s=(1,2), p=(0,1))  →(B,256,1,209)
        freq_down3  : Conv2d(256→256,k=(1,3),  s=(1,2), p=(0,1))  →(B,256,1,105)

    Bottleneck:
        res_block   : Conv2d(256→256,k=(1,3),p=(0,1)) ×2 + residual
        cond_add    : ConditioningMLP output broadcast-added

    Decoder:
        freq_up3    : ConvTranspose2d(512→256, k=(1,4), s=(1,2), p=(0,1), op=(0,1))
        freq_up2    : ConvTranspose2d(512→128, k=(1,4), s=(1,2), p=(0,1))
        freq_up1    : ConvTranspose2d(256→64,  k=(1,4), s=(1,2), p=(0,1))
        time_up     : ConvTranspose2d(128→2,   k=(11,1),s=(11,1))

    Args:
        in_channels:   Number of input channels (default 2).
        base_channels: Base channel width (default 64).
        cond_dim:      Conditioning vector length (default 64).
        cond_hidden:   Hidden dim of the conditioning MLP (default 128).
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        cond_dim: int = 64,
        cond_hidden: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.cond_dim = cond_dim
        self.cond_hidden = cond_hidden

        C = base_channels  # 64

        # ── Encoder ──────────────────────────────────────────────────────────
        # (B, 2, 11, 836) → (B, 64, 1, 836)
        self.time_down = nn.Conv2d(in_channels, C, kernel_size=(11, 1), stride=(11, 1))

        # (B, 64, 1, 836) → (B, 128, 1, 418)
        self.freq_down1 = nn.Conv2d(C, C * 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # (B, 128, 1, 418) → (B, 256, 1, 209)
        self.freq_down2 = nn.Conv2d(C * 2, C * 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # (B, 256, 1, 209) → (B, 256, 1, 105)
        self.freq_down3 = nn.Conv2d(C * 4, C * 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # ── Bottleneck residual block ─────────────────────────────────────────
        self.res_conv1 = nn.Conv2d(C * 4, C * 4, kernel_size=(1, 3), padding=(0, 1))
        self.res_conv2 = nn.Conv2d(C * 4, C * 4, kernel_size=(1, 3), padding=(0, 1))
        self.res_relu = nn.ReLU()

        # Conditioning MLP: 64 → 128 → 256 (bottleneck channels)
        self.cond_mlp = ConditioningMLP(in_dim=cond_dim, hidden_dim=cond_hidden, out_dim=C * 4)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Skip concat doubles input channels at each decoder stage.

        # (B, 512, 1, 105) → (B, 256, 1, 209)
        # output = (105-1)*2 - 2*1 + 3 = 209  ✓
        self.freq_up3 = nn.ConvTranspose2d(
            C * 8, C * 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
        )

        # (B, 512, 1, 209) → (B, 128, 1, 418)
        self.freq_up2 = nn.ConvTranspose2d(
            C * 8, C * 2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
        )

        # (B, 256, 1, 418) → (B, 64, 1, 836)
        self.freq_up1 = nn.ConvTranspose2d(
            C * 4, C, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
        )

        # (B, 128, 1, 836) → (B, 2, 11, 836)
        self.time_up = nn.ConvTranspose2d(
            C * 2, in_channels, kernel_size=(11, 1), stride=(11, 1)
        )

        self.relu = nn.ReLU()

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise for a noisy spectrogram window.

        Args:
            x:    Noisy input, shape ``(B, 2, 11, 836)``.
            cond: Conditioning vector, shape ``(B, 64)``.

        Returns:
            Predicted noise tensor of shape ``(B, 2, 11, 836)``.
        """
        validate_input_shape(x, (-1, self.in_channels, 11, 836), name="x")
        validate_input_shape(cond, (-1, self.cond_dim), name="cond")
        validate_finite(x, name="x")
        validate_finite(cond, name="cond")

        # ── Encoder ──────────────────────────────────────────────────────────
        # (B,2,11,836) → (B,64,1,836)
        s0 = self.relu(self.time_down(x))

        # (B,64,1,836) → (B,128,1,418)
        s1 = self.relu(self.freq_down1(s0))

        # (B,128,1,418) → (B,256,1,209)
        s2 = self.relu(self.freq_down2(s1))

        # (B,256,1,209) → (B,256,1,105)
        s3 = self.relu(self.freq_down3(s2))

        # ── Bottleneck ───────────────────────────────────────────────────────
        h = self.relu(self.res_conv1(s3))
        h = self.res_conv2(h)
        h = self.relu(h + s3)  # residual connection

        # Conditioning injection: (B,256) → (B,256,1,1) broadcast over freq
        c = self.cond_mlp(cond).unsqueeze(-1).unsqueeze(-1)  # (B,256,1,1)
        h = h + c

        # ── Decoder ──────────────────────────────────────────────────────────
        # (B,512,1,105) → (B,256,1,209)
        h = self.relu(self.freq_up3(torch.cat([h, s3], dim=1)))

        # (B,512,1,209) → (B,128,1,418)
        h = self.relu(self.freq_up2(torch.cat([h, s2], dim=1)))

        # (B,256,1,418) → (B,64,1,836)
        h = self.relu(self.freq_up1(torch.cat([h, s1], dim=1)))

        # (B,128,1,836) → (B,2,11,836)
        h = self.time_up(torch.cat([h, s0], dim=1))

        return h

    # ── Config ───────────────────────────────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        """Return constructor hyperparameters for model reconstruction.

        Returns:
            Dict with keys: in_channels, base_channels, cond_dim, cond_hidden.
        """
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "cond_dim": self.cond_dim,
            "cond_hidden": self.cond_hidden,
        }
