"""
ImageGenerator — 76 independent sub-U-Nets, one per 11×11 freq-band subwindow.

Full window shape:   (B, 2, 836, 11)  — freq × time
Subwindow shape:     (B, 2,  11, 11)  — 836 / 11 = 76 non-overlapping freq bands
Cond shape:          (B, 64)          — same token broadcast to all 76 sub-U-Nets
t shape:             (B,)             — diffusion timestep in [0, T-1]
Output shape:        (B, 2, 836, 11)  — reassembled from 76 subwindow predictions

Each SubUNet has:
  - Its own independent weight matrices (no sharing across bands)
  - A registered buffer holding the mean subtemplate for its freq band
  - Isotropic 3×3 replicate-padded convs (freq and time treated equally)
  - Sinusoidal t embedding + conditioning token injected at bottleneck
  - Skip connections preserving freq-position info within the band

Inference start:
  x = sqrt(ᾱ_{t_start}) * template + sqrt(1 - ᾱ_{t_start}) * N(0,1)
  where t_start is configurable (0 = pure template, T-1 = pure noise).
  template is split into 76 subtemplates stored per SubUNet.

DATA_STD and NUM_BANDS are module-level constants so downstream code can
import them without instantiating the model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

DATA_STD:  float = 4.2
NUM_BANDS: int   = 76   # 836 / 11 = 76 exactly
BAND_H:    int   = 11   # freq bins per subwindow
BAND_W:    int   = 11   # time frames per subwindow


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _gn(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class _SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / max(half - 1, 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        args = t[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


class _ResBlock(nn.Module):
    """3×3 replicate-padded conv block with t-embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1  = _gn(in_ch)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1, padding_mode='replicate')
        self.norm2  = _gn(out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.act    = nn.SiLU()
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Single sub-U-Net for one 11×11×2 freq-band subwindow
# ---------------------------------------------------------------------------

class SubUNet(nn.Module):
    """Independent U-Net for one freq-band subwindow (2, 11, 11).

    Architecture (C = base_channels):
      enc_in   : Conv2d(2 → C,   3×3 replicate)
      enc_res1 : ResBlock(C → C*2)   + down stride-(2,2)  →  (C*2, 5, 5)
      enc_res2 : ResBlock(C*2 → C*4) + down stride-(1,1) [same size bottleneck]
      bot_res  : ResBlock(C*4 → C*4)  ← t-emb + cond injected here
      up1      : ConvTranspose (2,2)   →  (C*4, 11, 11)  [output_padding=(1,1)]
      dec_res1 : ResBlock(C*4+C*2 → C*2)
      dec_res2 : ResBlock(C*2+C → C)
      out      : GN + SiLU + Conv2d(C → 2, 1×1)

    The subtemplate (mean of this band across the dataset) is stored as a
    registered buffer of shape (1, 2, 11, 11) for use at inference start.

    Args:
        base_channels: Channel width (default 32 — small since we have 76 of these).
        cond_dim:      Conditioning token length (default 64).
        t_dim:         Timestep embedding dimension (default 128).
    """

    def __init__(self, base_channels: int = 32, cond_dim: int = 64, t_dim: int = 128):
        super().__init__()
        C = base_channels

        # Subtemplate for this freq band — filled in by ImageGenerator.set_template()
        self.register_buffer("subtemplate", torch.zeros(1, 2, BAND_H, BAND_W))

        # Timestep embedding (each SubUNet has its own so weights are independent)
        self.time_emb = _SinusoidalEmb(t_dim)

        # Conditioning projection: 64 → C*4 (bottleneck width)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, C * 4)
        )

        # ── Encoder ───────────────────────────────────────────────────────
        # (B, 2, 11, 11) → (B, C, 11, 11)
        self.enc_in   = nn.Conv2d(2, C, 3, padding=1, padding_mode='replicate')

        # (B, C, 11, 11) → res → (B, C*2, 11, 11) → down → (B, C*2, 5, 5)
        self.enc_res1 = _ResBlock(C,     C * 2, t_dim)
        self.down1    = nn.Conv2d(C * 2, C * 2, kernel_size=2, stride=2)
        # 11 → floor(11/2) = 5

        # (B, C*2, 5, 5) → res → (B, C*4, 5, 5)  [no spatial change]
        self.enc_res2 = _ResBlock(C * 2, C * 4, t_dim)

        # ── Bottleneck at (B, C*4, 5, 5) ─────────────────────────────────
        self.bot_res  = _ResBlock(C * 4, C * 4, t_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        # (B, C*4, 5, 5) → up → (B, C*4, 11, 11)  output_padding=(1,1) restores 11
        self.up1      = nn.ConvTranspose2d(C * 4, C * 4, kernel_size=2, stride=2,
                                           output_padding=1)

        # skip from enc_res1 is (B, C*2, 11, 11) → cat → (B, C*4+C*2, 11, 11)
        self.dec_res1 = _ResBlock(C * 4 + C * 2, C * 2, t_dim)

        # skip from enc_in is (B, C, 11, 11) → cat → (B, C*2+C, 11, 11)
        self.dec_res2 = _ResBlock(C * 2 + C, C, t_dim)

        # Output
        self.out_norm = _gn(C)
        self.out_conv = nn.Conv2d(C, 2, 1)

    def forward(
        self,
        x:    torch.Tensor,   # (B, 2, 11, 11) — normalized
        cond: torch.Tensor,   # (B, 64)
        t:    torch.Tensor,   # (B,)
    ) -> torch.Tensor:        # (B, 2, 11, 11) — predicted noise, normalized
        t_emb = self.time_emb(t)          # (B, t_dim)
        c_emb = self.cond_proj(cond)      # (B, C*4)

        s0  = self.enc_in(x)                         # (B, C,   11, 11)
        s1  = self.enc_res1(s0, t_emb)               # (B, C*2, 11, 11)
        h   = self.down1(s1)                         # (B, C*2,  5,  5)
        h   = self.enc_res2(h, t_emb)                # (B, C*4,  5,  5)

        h   = self.bot_res(h, t_emb)
        h   = h + c_emb[:, :, None, None]            # inject conditioning

        h   = self.up1(h)                            # (B, C*4, 11, 11)
        h   = self.dec_res1(torch.cat([h, s1], 1), t_emb)   # (B, C*2, 11, 11)
        h   = self.dec_res2(torch.cat([h, s0], 1), t_emb)   # (B, C,   11, 11)

        return self.out_conv(F.silu(self.out_norm(h)))       # (B, 2,   11, 11)


# ---------------------------------------------------------------------------
# Full ImageGenerator: 76 independent SubUNets
# ---------------------------------------------------------------------------

class ImageGenerator(nn.Module):
    """76 independent SubUNets, one per 11-freq-bin band of the 836-bin axis.

    Internally normalizes by DATA_STD so all external tensors stay in raw
    spectrogram units (positive and negative floats, std ≈ 4.2).

    Args:
        in_channels:   Must be 2 (kept for config round-trip compatibility).
        base_channels: SubUNet base channel width (default 32).
        cond_dim:      Encoder token length (default 64).
        t_dim:         Timestep embedding dimension (default 128).
        num_steps:     Total DDPM timesteps T (default 1000).
    """

    def __init__(
        self,
        in_channels:   int = 2,
        base_channels: int = 32,
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

        self.register_buffer("data_std", torch.tensor(DATA_STD))

        # 76 fully independent SubUNets
        self.subnets = nn.ModuleList([
            SubUNet(base_channels=base_channels, cond_dim=cond_dim, t_dim=t_dim)
            for _ in range(NUM_BANDS)
        ])

    # ── Template management ───────────────────────────────────────────────

    def set_template(self, template: torch.Tensor) -> None:
        """Store the dataset mean spectrogram, split into 76 subtemplates.

        Args:
            template: Mean spectrogram, shape (2, 836, 11) or (1, 2, 836, 11),
                      in raw spectrogram units.
        """
        if template.dim() == 4:
            template = template.squeeze(0)   # (2, 836, 11)
        assert template.shape == (2, 836, 11), \
            f"template must be (2, 836, 11), got {tuple(template.shape)}"

        # Normalize once, store per-band
        t_norm = template / float(self.data_std)   # (2, 836, 11)
        for b, subnet in enumerate(self.subnets):
            band = t_norm[:, b * BAND_H : (b + 1) * BAND_H, :]  # (2, 11, 11)
            subnet.subtemplate.copy_(band.unsqueeze(0))           # (1, 2, 11, 11)

    def has_template(self) -> bool:
        """Return True if set_template() has been called with non-zero data."""
        return self.subnets[0].subtemplate.abs().sum().item() > 0.0

    # ── Forward (noise prediction) ────────────────────────────────────────

    def forward(
        self,
        x:    torch.Tensor,   # (B, 2, 836, 11)  raw units
        cond: torch.Tensor,   # (B, 64)
        t:    torch.Tensor,   # (B,)  int in [0, num_steps-1]
    ) -> torch.Tensor:        # (B, 2, 836, 11)  raw units
        """Predict noise ε. Each of 76 SubUNets handles its own freq band."""
        x_norm = x / self.data_std   # normalize to ~unit variance

        bands_out = []
        for b, subnet in enumerate(self.subnets):
            xb = x_norm[:, :, b * BAND_H : (b + 1) * BAND_H, :]  # (B, 2, 11, 11)
            eb = subnet(xb, cond, t)                                # (B, 2, 11, 11)
            bands_out.append(eb)

        eps_norm = torch.cat(bands_out, dim=2)   # (B, 2, 836, 11)
        return eps_norm * self.data_std          # back to raw units

    # ── Inference helpers ─────────────────────────────────────────────────

    def make_start(
        self,
        batch_size: int,
        t_start:    int,
        sqrt_ab:    torch.Tensor,   # (T,) from schedule
        sqrt_1mab:  torch.Tensor,   # (T,) from schedule
        device:     torch.device,
    ) -> torch.Tensor:
        """Build the starting canvas for reverse diffusion.

        x = sqrt(ᾱ_{t_start}) * template + sqrt(1-ᾱ_{t_start}) * N(0,1)

        If no template has been set, falls back to pure N(0, data_std²) noise.

        Args:
            batch_size: B.
            t_start:    Noise level index in [0, num_steps-1].
            sqrt_ab:    sqrt(ᾱ_t) schedule tensor, shape (T,).
            sqrt_1mab:  sqrt(1-ᾱ_t) schedule tensor, shape (T,).
            device:     Target device.

        Returns:
            Starting canvas in raw spectrogram units, shape (B, 2, 836, 11).
        """
        noise = torch.randn(batch_size, 2, 836, 11, device=device)

        if not self.has_template():
            # No template: pure noise scaled to data_std
            return noise * float(self.data_std)

        # Assemble full template from subtemplates (normalized space)
        parts = [subnet.subtemplate.expand(batch_size, -1, -1, -1)
                 for subnet in self.subnets]
        tmpl_norm = torch.cat(parts, dim=2)   # (B, 2, 836, 11) normalized

        s_ab   = sqrt_ab[t_start].item()
        s_1mab = sqrt_1mab[t_start].item()
        x_norm = s_ab * tmpl_norm + s_1mab * noise

        return x_norm * float(self.data_std)  # raw units

    # ── Config ────────────────────────────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_channels":   self.in_channels,
            "base_channels": self.base_channels,
            "cond_dim":      self.cond_dim,
            "t_dim":         self.t_dim,
            "num_steps":     self.num_steps,
        }
