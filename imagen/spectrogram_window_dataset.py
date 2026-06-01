"""
SpectrogramWindowDataset — overlapping (2, 836, 11) windows for DDPM training.

Window orientation: (2, 836, 11) — channels × freq × time.
This matches WhipstrEncoder's expected input (B, 2, 836, window_size).

Each __getitem__ returns:
  window : (2, 836, 11) float tensor  — raw spectrogram values (not normalized)
  token  : (64,)        float tensor  — frozen encoder output for that window
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union


def _load_spectrogram(flac_path: str) -> torch.Tensor:
    """Load an audio file into a (2, 836, W) spectrogram tensor."""
    from phase import Phase

    phase = Phase(y_reverse=True)
    audio = phase.to_tensor_flac(flac_path)

    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio).float()
    else:
        audio = audio.float()

    num_freqs = phase.num_freqs
    W = audio.shape[0] // num_freqs
    audio = audio[:W * num_freqs].reshape(W, num_freqs, 2).permute(2, 1, 0)
    # audio: (2, num_freqs, W)

    HEIGHT = 836
    channels, height, width = audio.shape
    if height < HEIGHT:
        pad = torch.zeros(channels, HEIGHT - height, width)
        audio = torch.cat([audio, pad], dim=1)
    elif height > HEIGHT:
        audio = audio[:, :HEIGHT, :]

    return audio.float().contiguous()  # (2, 836, W)


def _load_frozen_encoder(checkpoint_path: str, device: torch.device):
    """Load WhipstrEncoder from checkpoint, freeze, and return."""
    from whipstr.whipstr_encoder import WhipstrEncoder

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


class SpectrogramWindowDataset(Dataset):
    """Overlapping (2, 836, 11) spectrogram windows paired with encoder tokens.

    Window orientation is (channels, freq, time) = (2, 836, 11), matching
    WhipstrEncoder's expected input format.

    Tokens are precomputed once at init; windows are sliced lazily in __getitem__.

    Args:
        source:                  TSV path (str) or list of (2, 836, W) tensors.
        encoder_checkpoint_path: Path to frozen WhipstrEncoder checkpoint.
        limit:                   Max FLAC files to load from TSV (0 = all).
        device:                  Device for encoder inference.
    """

    WINDOW_TIME = 11   # time frames per window — matches encoder window_size
    STRIDE      = 1    # stride between windows
    COND_DIM    = 64   # encoder output dimension

    def __init__(
        self,
        source: Union[str, List[torch.Tensor]],
        encoder_checkpoint_path: str,
        limit: int = 0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # ── Load spectrograms ─────────────────────────────────────────────
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
            raise TypeError(f"source must be str or list of tensors, got {type(source).__name__}")

        # ── Load encoder ──────────────────────────────────────────────────
        self.encoder = _load_frozen_encoder(encoder_checkpoint_path, self.device)

        # ── Build index and precompute tokens ─────────────────────────────
        self._specs:  List[torch.Tensor] = []
        self._tokens: List[torch.Tensor] = []
        self._index:  List[Tuple[int, int]] = []  # (spec_idx, w_idx)

        for spec in spectrograms:
            spec = spec.float()
            if spec.ndim != 3 or spec.shape[0] != 2 or spec.shape[1] != 836:
                raise ValueError(
                    f"Each spectrogram must be (2, 836, W), got {tuple(spec.shape)}"
                )

            W = spec.shape[2]
            T = W - self.WINDOW_TIME + 1
            if T <= 0:
                raise ValueError(
                    f"Spectrogram width {W} too small for window size {self.WINDOW_TIME}"
                )

            # Precompute tokens: encoder expects (B, 2, 836, window_size)
            # Feed the full spectrogram at once — encoder slides internally
            with torch.no_grad():
                tokens = self.encoder(spec.unsqueeze(0).to(self.device))  # (1, T, 64)
            tokens = tokens.squeeze(0).cpu()  # (T, 64)

            self._specs.append(spec.cpu())
            self._tokens.append(tokens)
            for w in range(T):
                self._index.append((len(self._specs) - 1, w))

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self._index):
            raise IndexError(f"Index {idx} out of range (size {len(self._index)})")

        spec_idx, w_idx = self._index[idx]
        spec  = self._specs[spec_idx]                          # (2, 836, W)
        token = self._tokens[spec_idx][w_idx]                  # (64,)

        # Window: (2, 836, 11) — freq × time, matching encoder orientation
        window = spec[:, :, w_idx : w_idx + self.WINDOW_TIME].contiguous()

        return window, token
