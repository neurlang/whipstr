"""
Test that batch_size=2 training works with silence and white noise audio,
verifying phase spectrogram properties through WhipstrTSVSpeechDataset + DataLoader.
"""
import os
import tempfile
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
from whipstr.whipstr_encoder import WhipstrEncoder


def test_silence_and_noise_phase_spectrograms_with_batch_size_two():
    sample_rate = 44100
    duration = 1.0
    num_samples = int(sample_rate * duration)
    np.random.seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        silence_path = os.path.join(tmpdir, "silence.wav")
        sf.write(silence_path, np.zeros(num_samples, dtype=np.float64), sample_rate)

        noise_path = os.path.join(tmpdir, "noise.wav")
        sf.write(noise_path, np.random.randn(num_samples).astype(np.float64), sample_rate)

        tsv_path = os.path.join(tmpdir, "test.tsv")
        with open(tsv_path, "w") as f:
            f.write(f"{silence_path}\tsilence\n")
            f.write(f"{noise_path}\tnoise\n")

        dataset = WhipstrTSVSpeechDataset(tsv_path)
        dataloader = DataLoader(dataset, batch_size=2)

        images, transcriptions = next(iter(dataloader))
        # images shape: (2, 2, 836, W)

        silence_spec = images[0]
        noise_spec = images[1]

        assert torch.all(silence_spec == 0), "Silence phase spectrogram should be all zeros"
        assert not torch.all(noise_spec == 0), "White noise phase spectrogram should NOT be all zeros"
        assert silence_spec.shape == noise_spec.shape

        # Indirectly verify training with batch_size=2 works:
        # forward pass through encoder
        encoder = WhipstrEncoder(stride=1, window_size=28)
        encoder.eval()
        with torch.no_grad():
            encoder_output = encoder(images)

        assert encoder_output.shape[0] == 2
        assert encoder_output.shape[2] == 64
        assert encoder_output.dtype == torch.float32
