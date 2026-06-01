"""
ImagenTrainer — orchestrates Stage 2 diffusion training conditioned on encoder tokens.

Standard DDPM noise-prediction loss.  The 64-dim encoder token conditions the
U-Net via the bottleneck MLP.  No auxiliary reconstruction loss.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

from .image_generator import ImageGenerator
from .spectrogram_window_dataset import SpectrogramWindowDataset


class ImagenTrainer:
    """Trains an ImageGenerator with a diffusion noise-prediction objective.

    The generator takes a noisy 2D spectrogram window + a 64-dim conditioning
    token (from the frozen Stage 1 encoder), and predicts the noise that was
    added.  Loss is standard DDPM: MSE(noise_pred, noise).

    Args:
        encoder_checkpoint_path: Path to the WhipstrEncoder checkpoint (.pt).
        generator:               Pre-built ImageGenerator instance (optional).
                                 If None, a default ImageGenerator() is created.
        lr:                      Learning rate for the Adam optimizer.
        device:                  Torch device string (e.g. "cpu", "cuda").
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

        # ── Load and freeze WhipstrEncoder (for inference token computation) ─
        if not os.path.exists(encoder_checkpoint_path):
            raise FileNotFoundError(
                f"WhipstrEncoder checkpoint not found: {encoder_checkpoint_path}"
            )
        self.encoder = self._load_frozen_encoder(encoder_checkpoint_path)

        # ── ImageGenerator ───────────────────────────────────────────────────
        self.generator = generator if generator is not None else ImageGenerator()
        self.generator = self.generator.to(self.device)

        # ── Optimizer ────────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.generator.parameters(), lr=lr)

        self.mse = nn.MSELoss()

    # ── Encoder loading ──────────────────────────────────────────────────────

    def _load_frozen_encoder(self, path: str):
        """Load WhipstrEncoder from checkpoint and freeze all parameters."""
        from whipstr.whipstr_encoder import WhipstrEncoder

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
            state_dict = ckpt["encoder_state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        encoder = WhipstrEncoder(window_size=11)
        encoder.load_state_dict(state_dict)
        encoder = encoder.to(self.device)

        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

        return encoder

    # ── Noise injection ──────────────────────────────────────────────────────

    def add_noise(
        self, x_clean: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise scaled by noise level t.

        Args:
            x_clean: Clean spectrogram windows, shape (B, 2, 11, 836).
            t:       Noise level tensor, shape (B,) or scalar, values in [0, 1].

        Returns:
            (x_t, noise): Noisy input and the noise that was added.
        """
        noise = torch.randn_like(x_clean)
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        x_t = x_clean + t * noise
        return x_t, noise

    # ── Loss computation ────────────────────────────────────────────────────

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard DDPM noise-prediction loss.

        Args:
            x_clean: Clean windows, shape (B, 2, 11, 836).
            cond:    Conditioning tokens from encoder, shape (B, 64).

        Returns:
            Scalar loss tensor.
        """
        B = x_clean.shape[0]
        t = torch.rand(B, device=self.device)

        x_t, noise = self.add_noise(x_clean, t)

        noise_pred = self.generator(x_t, cond)
        loss = self.mse(noise_pred, noise)

        return loss

    # ── Training loop ────────────────────────────────────────────────────────

    def train(
        self,
        dataset: SpectrogramWindowDataset,
        num_epochs: int,
        batch_size: int = 16,
        checkpoint_interval: int = 5,
        output_dir: str = "checkpoints",
    ) -> None:
        """Run the training loop.

        Args:
            dataset:             SpectrogramWindowDataset instance.
            num_epochs:          Total number of epochs to train.
            batch_size:          DataLoader batch size.
            checkpoint_interval: Save a checkpoint every N epochs.
            output_dir:          Directory to write checkpoint files.
        """
        os.makedirs(output_dir, exist_ok=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.generator.train()
            total_loss = 0.0
            n_batches = 0

            for windows, cond in loader:
                windows = windows.to(self.device)
                cond = cond.to(self.device)

                self.optimizer.zero_grad()
                loss = self.compute_loss(windows, cond)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            print(
                f"Epoch {epoch + 1}/{self.current_epoch + num_epochs} — "
                f"loss: {avg_loss:.6f}"
            )

            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1, avg_loss, output_dir)

        self.current_epoch += num_epochs

    # ── Checkpoint save/load ─────────────────────────────────────────────────

    def _save_checkpoint(
        self, epoch: int, loss: float, output_dir: str
    ) -> str:
        """Save a training checkpoint and return the file path."""
        path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": self.generator.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "config": self.generator.get_config(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint.

        Restores generator weights, optimizer state, and epoch counter.

        Args:
            checkpoint_path: Path to a .pt checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.generator.load_state_dict(ckpt["generator_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]

        print(f"Resumed from checkpoint: {checkpoint_path} (epoch {self.current_epoch})")
