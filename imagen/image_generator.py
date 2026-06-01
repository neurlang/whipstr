"""
ImageGenerator — U-Net noise predictor for DDPM on spectrogram windows.

Input shape:  (B, 2, 836, 11)  — noisy spectrogram window, freq × time
Cond shape:   (B, 64)          — encoder token
t shape:      (B,)             — diffusion timestep in [0, T-1]
Output shape: (B, 2, 836, 11)  — predicted noise ε

Freq axis: 836 = 2 × 2 × 11 × 19  →  exactly 2 clean stride-2 halvings:
  836 → 418 → 209  (bottleneck)
  No third downsample; 209 is odd and cannot be halved cleanly.

Time axis (11): never downsampled — too narrow, and temporal structure matters.

Architecture
------------
  enc_in   : Conv2d(2 → C,   3×3)
  enc_res1 : ResBlock(C   → C*2)  +  down1: stride-(2,1) on freq  →  418
  enc_res2 : ResBlock(C*2 → C*4)  +  down2: stride-(2,1) on freq  →  209
  bot_res1 : ResBlock(C*4 → C*4)  — conditioning injected here
  bot_res2 : ResBlock(C*4 → C*4)
  up2      : ConvTranspose stride-(2,1)  →  418
  dec_res2 : ResBlock(C*8 → C*2)  (skip from enc_res2)
  up1      : ConvTranspose stride-(2,1)  →  836
  dec_res1 : ResBlock(C*4 → C)    (skip from enc_res1)
  out      : GroupNorm + SiLU + Conv2d(C → 2, 1×1)

All skip connections are exact-size — no cropping or padding required.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# Empirical std of phase-spectrogram values across the training corpus.
# Baked in so training, inference, and evaluation all normalize identically.
DATA_STD: float = 4.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _groupnorm(channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 32 groups, always divides channels evenly."""
    groups = min(32, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding → two-layer MLP.

    Args:
        dim: Output embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1))
        self.register_buffer("freqs", freqs)  # (half,)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t (B,) int.  Returns: (B, dim)."""
        t = t.float()
        args = t[:, None] * self.freqs[None, :]           # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Two 3×3 convs with GroupNorm + SiLU and timestep-embedding injection.

    Uses replicate padding so the narrow time axis (width=11) is not corrupted
    by zero-padding artifacts at the left/right edges.

    Args:
        in_ch:  Input channels.
        out_ch: Output channels.
        t_dim:  Timestep embedding dimension.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1  = _groupnorm(in_ch)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1, padding_mode='replicate')
        self.norm2  = _groupnorm(out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.act    = nn.SiLU()
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]  # inject t
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ImageGenerator(nn.Module):
    """U-Net noise predictor for DDPM on (2, 836, 11) spectrogram windows.

    Internally normalizes by DATA_STD so all external tensors stay in raw
    spectrogram units (positive and negative floats).

    Args:
        in_channels:   Spectrogram channels (default 2).
        base_channels: Base channel width (default 64).
        cond_dim:      Encoder token length (default 64).
        t_dim:         Timestep embedding dimension (default 128).
        num_steps:     Total DDPM timesteps T (default 1000).
    """

    def __init__(
        self,
        in_channels:   int = 2,
        base_channels: int = 64,
        cond_dim:      int = 64,
        t_dim:         int = 128,
        num_steps:     int = 1000,
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.base_channels = base_channels
        self.cond_dim      = cond_dim
        self.t_dim         = t_dim
        self.num_steps     = num_steps

        C = base_channels  # 64

        # Normalization constant — baked in, not learned
        self.register_buffer("data_std", torch.tensor(DATA_STD))

        # ── Timestep embedding ────────────────────────────────────────────
        self.time_emb = SinusoidalTimeEmbedding(t_dim)

        # ── Conditioning projection ───────────────────────────────────────
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, C * 4),
        )

        # ── Encoder ──────────────────────────────────────────────────────
        # (B, 2, 836, 11) → (B, C, 836, 11)
        self.enc_in   = nn.Conv2d(in_channels, C, 3, padding=1, padding_mode='replicate')

        # (B, C, 836, 11) → (B, C*2, 836, 11) → down → (B, C*2, 418, 11)
        self.enc_res1 = ResBlock(C,     C * 2, t_dim)
        self.down1    = nn.Conv2d(C * 2, C * 2, kernel_size=(2, 1), stride=(2, 1))

        # (B, C*2, 418, 11) → (B, C*4, 418, 11) → down → (B, C*4, 209, 11)
        self.enc_res2 = ResBlock(C * 2, C * 4, t_dim)
        self.down2    = nn.Conv2d(C * 4, C * 4, kernel_size=(2, 1), stride=(2, 1))

        # ── Bottleneck at (B, C*4, 209, 11) ──────────────────────────────
        self.bot_res1 = ResBlock(C * 4, C * 4, t_dim)
        self.bot_res2 = ResBlock(C * 4, C * 4, t_dim)

        # ── Decoder ──────────────────────────────────────────────────────
        # (B, C*4, 209, 11) → up → (B, C*4, 418, 11)
        self.up2      = nn.ConvTranspose2d(C * 4, C * 4, kernel_size=(2, 1), stride=(2, 1))
        # skip from enc_res2: (B, C*4, 418, 11) → cat → (B, C*8, 418, 11)
        self.dec_res2 = ResBlock(C * 8, C * 2, t_dim)

        # (B, C*2, 418, 11) → up → (B, C*2, 836, 11)
        self.up1      = nn.ConvTranspose2d(C * 2, C * 2, kernel_size=(2, 1), stride=(2, 1))
        # skip from enc_res1: (B, C*2, 836, 11) → cat → (B, C*4, 836, 11)
        self.dec_res1 = ResBlock(C * 4, C, t_dim)

        # Output projection
        self.out_norm = _groupnorm(C)
        self.out_conv = nn.Conv2d(C, in_channels, 1)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x:    torch.Tensor,
        cond: torch.Tensor,
        t:    torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise ε for a noisy spectrogram window.

        Args:
            x:    Noisy window, raw spectrogram units, shape (B, 2, 836, 11).
            cond: Encoder token, shape (B, 64).
            t:    Integer timestep, shape (B,), values in [0, num_steps-1].

        Returns:
            Predicted noise ε, raw spectrogram units, shape (B, 2, 836, 11).
        """
        # Normalize to ~unit variance for stable diffusion arithmetic
        x = x / self.data_std

        # Embeddings
        t_emb = self.time_emb(t)        # (B, t_dim)
        c_emb = self.cond_proj(cond)    # (B, C*4)

        # ── Encoder ──────────────────────────────────────────────────────
        h0  = self.enc_in(x)                    # (B, C,   836, 11)
        s1  = self.enc_res1(h0,  t_emb)         # (B, C*2, 836, 11)  ← skip
        h1d = self.down1(s1)                    # (B, C*2, 418, 11)

        s2  = self.enc_res2(h1d, t_emb)         # (B, C*4, 418, 11)  ← skip
        h2d = self.down2(s2)                    # (B, C*4, 209, 11)

        # ── Bottleneck ────────────────────────────────────────────────────
        h = self.bot_res1(h2d, t_emb)
        h = h + c_emb[:, :, None, None]         # inject conditioning
        h = self.bot_res2(h,   t_emb)           # (B, C*4, 209, 11)

        # ── Decoder ──────────────────────────────────────────────────────
        h = self.up2(h)                          # (B, C*4, 418, 11)
        h = self.dec_res2(torch.cat([h, s2], dim=1), t_emb)  # (B, C*2, 418, 11)

        h = self.up1(h)                          # (B, C*2, 836, 11)
        h = self.dec_res1(torch.cat([h, s1], dim=1), t_emb)  # (B, C,   836, 11)

        # Output — no activation before final conv so output is unbounded (±)
        h = self.out_conv(F.silu(self.out_norm(h)))  # (B, 2, 836, 11)

        # Denormalize back to raw spectrogram units
        return h * self.data_std

    # ── Config ───────────────────────────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_channels":   self.in_channels,
            "base_channels": self.base_channels,
            "cond_dim":      self.cond_dim,
            "t_dim":         self.t_dim,
            "num_steps":     self.num_steps,
        }
