"""
ImagenTrainer — orchestrates Stage 2 diffusion training.

Loads a frozen WhipstrEncoder as the decoder, trains an ImageGenerator
with a noise-prediction + auxiliary reconstruction loss.

Requirements: 3.1, 3.2, 4.1–4.4, 5.1–5.4, 7.2
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
    """Trains an ImageGenerator using a diffusion-style noise-prediction objective.

    The frozen Stage 1 WhipstrEncoder acts as the decoder: its token output
    provides an auxiliary reconstruction loss that anchors the generator to
    acoustically meaningful reconstructions.

    Args:
        encoder_checkpoint_path: Path to the WhipstrEncoder checkpoint (.pt).
        generator:               Pre-built ImageGenerator instance (optional).
                                 If None, a default ImageGenerator() is created.
        lambda_recon:            Weight for the auxiliary reconstruction loss.
        lr:                      Learning rate for the Adam optimizer.
        device:                  Torch device string (e.g. "cpu", "cuda").
    """

    def __init__(
        self,
        encoder_checkpoint_path: str,
        generator: Optional[ImageGenerator] = None,
        lambda_recon: float = 0.1,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.lambda_recon = lambda_recon
        self.current_epoch = 0

        # ── Load and freeze WhipstrEncoder (Requirements 3.1, 3.2) ──────────
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

        # Support both raw state_dict and wrapped checkpoint dicts
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

        # Freeze all parameters (Requirement 3.2)
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

        return encoder

    # ── Noise injection (Requirement 4.1) ────────────────────────────────────

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
        # Broadcast t to (B, 1, 1, 1) for element-wise scaling
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        x_t = x_clean + t * noise
        return x_t, noise

    # ── Loss computation (Requirements 4.2, 4.3, 4.4) ────────────────────────

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute noise-prediction + auxiliary reconstruction loss.

        Args:
            x_clean: Clean windows, shape (B, 2, 11, 836).
            cond:    Conditioning vectors, shape (B, 64).

        Returns:
            (total_loss, loss_noise, loss_recon)
        """
        B = x_clean.shape[0]
        t = torch.rand(B, device=self.device)

        x_t, noise = self.add_noise(x_clean, t)

        # Noise prediction
        noise_pred = self.generator(x_t, cond)
        loss_noise = self.mse(noise_pred, noise)

        # Auxiliary reconstruction loss via frozen encoder (Requirement 4.3)
        x_denoised = x_t - noise_pred

        # WhipstrEncoder expects (B, 2, 836, W); our windows are (B, 2, 11, 836)
        # Permute to (B, 2, 836, 11) to match encoder's expected format
        x_denoised_enc = x_denoised.permute(0, 1, 3, 2)
        x_clean_enc = x_clean.permute(0, 1, 3, 2)

        with torch.no_grad():
            tokens_pred = self.encoder(x_denoised_enc)
            tokens_clean = self.encoder(x_clean_enc)

        loss_recon = self.mse(tokens_pred, tokens_clean)
        total_loss = loss_noise + self.lambda_recon * loss_recon

        return total_loss, loss_noise, loss_recon

    # ── Training loop (Requirements 5.1, 5.2, 5.3) ───────────────────────────

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
            total_noise = 0.0
            total_recon = 0.0
            n_batches = 0

            for windows, cond in loader:
                windows = windows.to(self.device)
                cond = cond.to(self.device)

                self.optimizer.zero_grad()
                loss, loss_noise, loss_recon = self.compute_loss(windows, cond)
                loss.backward()
                self.optimizer.step()

                total_noise += loss_noise.item()
                total_recon += loss_recon.item()
                n_batches += 1

            avg_noise = total_noise / max(n_batches, 1)
            avg_recon = total_recon / max(n_batches, 1)

            # Logging (Requirement 5.2)
            print(
                f"Epoch {epoch + 1}/{self.current_epoch + num_epochs} — "
                f"loss_noise: {avg_noise:.6f}  loss_recon: {avg_recon:.6f}"
            )

            # Checkpointing (Requirement 5.3)
            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1, avg_noise, avg_recon, output_dir)

        self.current_epoch += num_epochs

    # ── Checkpoint save/load (Requirements 5.3, 5.4, 7.2) ───────────────────

    def _save_checkpoint(
        self, epoch: int, loss_noise: float, loss_recon: float, output_dir: str
    ) -> str:
        """Save a training checkpoint and return the file path."""
        path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": self.generator.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_noise": loss_noise,
                "loss_recon": loss_recon,
                "config": self.generator.get_config(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint (Requirement 5.4).

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
