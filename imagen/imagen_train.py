"""
ImagenTrainer — DDPM training for the ImageGenerator acoustic codec decoder.

Noise schedule: linear beta schedule, standard DDPM ε-prediction.
  forward:  x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,   ε ~ N(0, I)
  loss:     MSE(ε_pred, ε)

The generator receives (x_t, cond, t) and predicts ε.
The encoder is frozen throughout.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from .image_generator import ImageGenerator
from .spectrogram_window_dataset import SpectrogramWindowDataset


def _make_linear_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear beta schedule → returns (sqrt_alpha_bar, sqrt_one_minus_alpha_bar).

    Both tensors have shape (num_steps,).
    """
    betas      = torch.linspace(beta_start, beta_end, num_steps)          # (T,)
    alphas     = 1.0 - betas                                               # (T,)
    alpha_bar  = torch.cumprod(alphas, dim=0)                              # (T,)
    sqrt_ab    = alpha_bar.sqrt()                                          # (T,)
    sqrt_1mab  = (1.0 - alpha_bar).sqrt()                                  # (T,)
    return sqrt_ab, sqrt_1mab


class ImagenTrainer:
    """Trains ImageGenerator with standard DDPM ε-prediction loss.

    Args:
        encoder_checkpoint_path: Path to the frozen WhipstrEncoder checkpoint.
        generator:               Pre-built ImageGenerator (created if None).
        lr:                      Adam learning rate.
        device:                  Torch device string.
    """

    def __init__(
        self,
        encoder_checkpoint_path: str,
        generator: Optional[ImageGenerator] = None,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.current_epoch = 0

        # ── Frozen encoder ────────────────────────────────────────────────
        self.encoder = self._load_frozen_encoder(encoder_checkpoint_path)

        # ── Generator ────────────────────────────────────────────────────
        self.generator = (generator if generator is not None else ImageGenerator()).to(self.device)

        # ── Noise schedule ────────────────────────────────────────────────
        T = self.generator.num_steps
        sqrt_ab, sqrt_1mab = _make_linear_schedule(T)
        self.register_schedule(sqrt_ab, sqrt_1mab)

        # ── Optimizer + loss ──────────────────────────────────────────────
        self.optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def register_schedule(self, sqrt_ab: torch.Tensor, sqrt_1mab: torch.Tensor) -> None:
        """Move schedule tensors to device (called after device changes too)."""
        self.sqrt_ab    = sqrt_ab.to(self.device)     # (T,)
        self.sqrt_1mab  = sqrt_1mab.to(self.device)   # (T,)

    # ── Encoder loading ──────────────────────────────────────────────────

    def _load_frozen_encoder(self, path: str):
        from whipstr.whipstr_encoder import WhipstrEncoder

        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict):
            state_dict = (
                ckpt.get("encoder_state_dict")
                or ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt
            )
        else:
            state_dict = ckpt

        encoder = WhipstrEncoder(window_size=11)
        encoder.load_state_dict(state_dict)
        encoder.to(self.device)
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        return encoder

    # ── DDPM forward process ─────────────────────────────────────────────

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t | x_0) = N(sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I).

        Args:
            x0: Clean windows, shape (B, 2, 836, 11).
            t:  Integer timestep indices, shape (B,).

        Returns:
            (x_t, noise): Both shape (B, 2, 836, 11).
        """
        noise = torch.randn_like(x0)
        s_ab   = self.sqrt_ab[t].view(-1, 1, 1, 1)    # (B,1,1,1)
        s_1mab = self.sqrt_1mab[t].view(-1, 1, 1, 1)
        x_t = s_ab * x0 + s_1mab * noise
        return x_t, noise

    # ── Loss ─────────────────────────────────────────────────────────────

    def compute_loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """DDPM ε-prediction loss.

        Args:
            x0:   Clean windows, shape (B, 2, 836, 11).
            cond: Encoder tokens, shape (B, 64).

        Returns:
            Scalar MSE loss.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.generator.num_steps, (B,), device=self.device)
        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.generator(x_t, cond, t)
        return self.mse(noise_pred, noise)

    # ── Training loop ────────────────────────────────────────────────────

    def train(
        self,
        dataset: SpectrogramWindowDataset,
        num_epochs: int,
        batch_size: int = 16,
        checkpoint_interval: int = 5,
        output_dir: str = "checkpoints/imagen",
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.generator.train()
            total_loss, n_batches = 0.0, 0

            for windows, cond in loader:
                windows = windows.to(self.device)   # (B, 2, 836, 11)
                cond    = cond.to(self.device)       # (B, 64)

                self.optimizer.zero_grad()
                loss = self.compute_loss(windows, cond)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            avg = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch + 1}/{self.current_epoch + num_epochs} — loss: {avg:.6f}")

            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1, avg, output_dir)

        self.current_epoch += num_epochs

    # ── Checkpoint ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, loss: float, output_dir: str) -> str:
        path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(
            {
                "epoch":                epoch,
                "generator_state_dict": self.generator.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss":                 loss,
                "config":               self.generator.get_config(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {self.current_epoch}")
