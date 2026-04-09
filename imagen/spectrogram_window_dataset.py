"""
SpectrogramWindowDataset — slices full (2, 836, W) spectrograms into
non-overlapping (2, 11, 836) windows for Stage 2 training.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import List, Union


def _load_spectrogram(flac_path: str) -> torch.Tensor:
    """Top-level function (picklable) for multiprocessing. Loads a single FLAC file."""
    import torch
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


class SpectrogramWindowDataset(Dataset):
    """
    Slices full spectrograms into non-overlapping (2, 11, 836) windows.

    Accepts either:
      - A list of (2, 836, W) float tensors, or
      - A TSV path string (delegates loading to WhipstrTSVSpeechDataset)

    Each __getitem__ returns (window, cond_vec) where:
      - window  : torch.Tensor of shape (2, 11, 836)
      - cond_vec: torch.Tensor of shape (64,)
                  [position_norm, mean_ch0, std_ch0, mean_ch1, std_ch1, 0…0]
    """

    WINDOW_TIME = 11   # time frames per window
    COND_DIM    = 64   # conditioning vector length

    def __init__(self, source: Union[str, List[torch.Tensor]], limit=0):
        """
        Args:
            source: TSV file path (str) or list of (2, 836, W) tensors.
        """
        if isinstance(source, str):
            from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
            tsv_ds = WhipstrTSVSpeechDataset(source, limit=limit)
            flac_paths = [tsv_ds.samples[i][0] for i in range(len(tsv_ds))]

            # First file must be loaded serially to initialise Phase state
            first = _load_spectrogram(flac_paths[0])
            rest = flac_paths[1:]

            if rest:
                # Use fork context on Linux: workers inherit the process image
                # instantly without re-importing everything (much faster than spawn)
                mp_ctx = get_context("fork")
                with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=mp_ctx) as pool:
                    remaining = list(pool.map(_load_spectrogram, rest, chunksize=4))
            else:
                remaining = []

            spectrograms = [first] + remaining
        elif isinstance(source, list):
            spectrograms = source
        else:
            raise TypeError(
                f"source must be a str (TSV path) or list of tensors, got {type(source).__name__}"
            )

        # Build flat index: list of (spectrogram_tensor, window_index, total_windows)
        self._windows: List[tuple] = []
        for spec in spectrograms:
            spec = spec.float()
            if spec.ndim != 3 or spec.shape[0] != 2 or spec.shape[1] != 836:
                raise ValueError(
                    f"Each spectrogram must have shape (2, 836, W), got {tuple(spec.shape)}"
                )
            W = spec.shape[2]
            n_windows = W // self.WINDOW_TIME
            if n_windows == 0:
                raise ValueError(
                    f"Spectrogram width {W} is too small to extract even one window of {self.WINDOW_TIME} frames"
                )
            for w_idx in range(n_windows):
                self._windows.append((spec, w_idx, n_windows))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self._windows):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._windows)}")

        spec, w_idx, n_windows = self._windows[idx]

        # Slice time frames: spec is (2, 836, W); time axis is dim-2
        t_start = w_idx * self.WINDOW_TIME
        t_end   = t_start + self.WINDOW_TIME
        # slice: (2, 836, 11)
        raw = spec[:, :, t_start:t_end]
        # transpose to (2, 11, 836)
        window = raw.permute(0, 2, 1).contiguous()

        # Build conditioning vector (64 floats)
        cond = self._build_cond(window, w_idx, n_windows)

        return window, cond

    # ------------------------------------------------------------------
    def _build_cond(self, window: torch.Tensor, w_idx: int, n_windows: int) -> torch.Tensor:
        """
        Build a 64-float conditioning vector for a window.

        Layout:
          [0]  position_norm  = w_idx / max(n_windows - 1, 1)
          [1]  mean_ch0
          [2]  std_ch0
          [3]  mean_ch1
          [4]  std_ch1
          [5..63] zeros
        """
        cond = torch.zeros(self.COND_DIM, dtype=torch.float32)
        cond[0] = w_idx / max(n_windows - 1, 1)
        cond[1] = window[0].mean()
        cond[2] = window[0].std()
        cond[3] = window[1].mean()
        cond[4] = window[1].std()
        return cond
