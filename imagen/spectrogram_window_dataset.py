"""
SpectrogramWindowDataset — slices full (2, 836, W) spectrograms into
overlapping (2, 11, 836) windows with stride=1 for Stage 2 training.

Each sample is (window, token) where:
  - window : overlapping spectrogram slice (2, 11, 836)
  - token  : 64-dim encoder output for that window (64,)

Tokens are precomputed during __init__ to avoid repeated encoder passes.
Windows are sliced lazily in __getitem__ to avoid RAM duplication.
"""

import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union


def _load_spectrogram(flac_path: str) -> torch.Tensor:
    """Load a single FLAC file into a (2, 836, W) spectrogram tensor."""
    from phase import Phase

    phase = Phase(y_reverse=True)
    audio_array = phase.to_tensor_flac(flac_path)

    if not isinstance(audio_array, torch.Tensor):
        audio_tensor = torch.from_numpy(audio_array).float()
    else:
        audio_tensor = audio_array.float()

    num_freqs = phase.num_freqs
    total_samples = audio_tensor.shape[0]
    actual_frames = total_samples // num_freqs
    audio_tensor = audio_tensor.reshape(actual_frames, num_freqs, 2).permute(2, 1, 0)

    channels, height, width = audio_tensor.shape
    if height < 836:
        pad = torch.zeros(channels, 836 - height, width)
        audio_tensor = torch.cat([audio_tensor, pad], dim=1)
    elif height > 836:
        audio_tensor = audio_tensor[:, :836, :]

    return audio_tensor.float()


def _load_frozen_encoder(checkpoint_path: str, device):
    """Load WhipstrEncoder from checkpoint, freeze, and return."""
    from whipstr.whipstr_encoder import WhipstrEncoder

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    encoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    return encoder


class SpectrogramWindowDataset(Dataset):
    """
    Slices full spectrograms into overlapping (2, 11, 836) windows with stride=1.

    Each window is paired with the 64-dim token produced by the frozen encoder
    when applied to that window.  Tokens are precomputed once; windows are
    sliced lazily to avoid RAM duplication.

    Accepts either:
      - A list of (2, 836, W) float tensors, or
      - A TSV path string (delegates loading to WhipstrTSVSpeechDataset)

    Each __getitem__ returns (window, token) where:
      - window : torch.Tensor of shape (2, 11, 836)
      - token  : torch.Tensor of shape (64,)
    """

    WINDOW_TIME = 11   # time frames per window (matches encoder window_size)
    STRIDE      = 1    # overlap stride (matches encoder stride)
    COND_DIM    = 64   # token dimension

    def __init__(
        self,
        source: Union[str, List[torch.Tensor]],
        encoder_checkpoint_path: str,
        limit: int = 0,
        device: str = "cpu",
    ):
        """
        Args:
            source: TSV file path (str) or list of (2, 836, W) tensors.
            encoder_checkpoint_path: Path to a WhipstrEncoder checkpoint.
            limit: Maximum number of FLAC files to load from TSV.
            device: Torch device for encoder inference.
        """
        self.device = torch.device(device)

        # ── Load spectrograms (serial, with progress bar) ──────────────────
        if isinstance(source, str):
            from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset

            tsv_ds = WhipstrTSVSpeechDataset(source, limit=limit)
            flac_paths = [tsv_ds.samples[i][0] for i in range(len(tsv_ds))]

            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(flac_paths), desc="Loading spectrograms")
            except ImportError:
                pbar = None

            spectrograms = []
            for fp in flac_paths:
                spectrograms.append(_load_spectrogram(fp))
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()
        elif isinstance(source, list):
            spectrograms = source
        else:
            raise TypeError(
                f"source must be a str (TSV path) or list of tensors, "
                f"got {type(source).__name__}"
            )

        # ── Load and freeze encoder ────────────────────────────────────────
        self.encoder = _load_frozen_encoder(encoder_checkpoint_path, self.device)

        # ── Build index and precompute tokens ──────────────────────────────
        self._specs: List[torch.Tensor] = []
        self._tokens: List[torch.Tensor] = []
        self._index: List[Tuple[int, int]] = []  # (spec_idx, w_idx)

        for spec in spectrograms:
            spec = spec.float().to(self.device)
            if spec.ndim != 3 or spec.shape[0] != 2 or spec.shape[1] != 836:
                raise ValueError(
                    f"Each spectrogram must have shape (2, 836, W), "
                    f"got {tuple(spec.shape)}"
                )

            W = spec.shape[2]
            T = W - self.WINDOW_TIME + 1  # overlapping windows with stride=1
            if T <= 0:
                raise ValueError(
                    f"Spectrogram width {W} is too small to extract even one "
                    f"window of {self.WINDOW_TIME} frames"
                )

            self._specs.append(spec)

            # Precompute tokens: feed entire spectrogram → encoder outputs (1, T, 64)
            with torch.no_grad():
                tokens = self.encoder(spec.unsqueeze(0))  # (1, T, 64)
            self._tokens.append(tokens.squeeze(0).cpu())  # (T, 64) → host

            for w_idx in range(T):
                self._index.append((len(self._specs) - 1, w_idx))

        # Move specs back to host to free GPU memory
        for i in range(len(self._specs)):
            self._specs[i] = self._specs[i].cpu()

    # ----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self._index):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self._index)}"
            )

        spec_idx, w_idx = self._index[idx]
        spec = self._specs[spec_idx]
        token = self._tokens[spec_idx][w_idx]  # (64,)

        # Slice overlapping window: spec is (2, 836, W); time axis is dim-2
        window_raw = spec[:, :, w_idx:w_idx + self.WINDOW_TIME]  # (2, 836, 11)
        window = window_raw.permute(0, 2, 1).contiguous()  # (2, 11, 836)

        return window, token
