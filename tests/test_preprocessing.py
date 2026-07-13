"""Golden test: validate HF feature extractor matches phase.Phase output."""

import math

import numpy as np
import phase
import torch

from whipstr.hf_integration import WhipstrFeatureExtractor


def _native_phase_spectrogram(waveform, sample_rate):
    """Replicate the dataset preprocessing pipeline via phase.Phase."""
    p = phase.Phase(y_reverse=True)
    p.reconfigure_sr(sample_rate)

    phase_flat = p.to_phase(waveform)  # (total_samples, 2)

    num_freqs = p.num_freqs
    total = phase_flat.shape[0]
    actual_frames = total // num_freqs

    tensor = torch.from_numpy(phase_flat).float()
    tensor = tensor.reshape(actual_frames, num_freqs, 2)
    tensor = tensor.permute(2, 1, 0)  # (2, num_freqs, W)

    if tensor.shape[1] < 836:
        pad = torch.zeros(2, 836 - tensor.shape[1], tensor.shape[2])
        tensor = torch.cat([tensor, pad], dim=1)
    elif tensor.shape[1] > 836:
        tensor = tensor[:, :836, :]

    return tensor


def _sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * math.pi * freq * t).astype(np.float64)


class TestPreprocessingMatchesPhase:
    """Both paths must produce identical spectrograms from the same waveform."""

    def _compare(self, waveform, sample_rate):
        native = _native_phase_spectrogram(waveform, sample_rate)

        fe = WhipstrFeatureExtractor(sampling_rate=sample_rate)
        hf_out = fe(waveform, sampling_rate=sample_rate)
        hf = hf_out["input_features"].squeeze(0)

        assert native.shape == hf.shape, f"Shape mismatch: {native.shape} vs {hf.shape}"
        torch.testing.assert_close(native, hf, rtol=0, atol=0)

    def test_48khz_440hz(self):
        self._compare(_sine_wave(440, 48000, 0.2), 48000)

    def test_48khz_silence(self):
        self._compare(np.zeros(int(48000 * 0.2), dtype=np.float64), 48000)

    def test_44khz_440hz(self):
        self._compare(_sine_wave(440, 44100, 0.2), 44100)

    def test_short_audio(self):
        self._compare(_sine_wave(440, 48000, 0.01), 48000)
