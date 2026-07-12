"""
Hugging Face integration for Whipstr STT (ASR) model.

Provides:
- HF-compatible model wrapper (WhipstrForConditionalGeneration)
- Tokenizer from your model.json vocab
- Feature extractor for audio preprocessing

Usage:

    # 1. Convert existing checkpoint to HF format
    uv run --with phase-spectrogram --with numpy --with torch --with scipy --with transformers python3 -m whipstr.hf_integration --checkpoint checkpoints/best_model.pt --model-json models/model.json
    
    # 2. Upload to Hugging Face Hub
    huggingface-cli login
    huggingface-cli upload ./hf_whipstr neuralang/en-whipstr-base-48khz-libritts-r
    
    # 3. Use with pipeline (after uploading)
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", 
                    model="neuralang/en-whipstr-base-48khz-libritts-r",
                    trust_remote_code=True)
    transcription = pipe("audio.wav")

"""

import json
import os

import numpy as np
import torch
from scipy import signal
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
    FeatureExtractionMixin,
)

from .whipstr_encoder import WhipstrEncoder
from .whipstr_transformer import WhipstrTransformer


class WhipstrConfig(PretrainedConfig):
    model_type = "whipstr"
    
    def __init__(
        self, vocab_size=None, stride=1, window_size=11,
        d_model=256, nhead=8, num_encoder_layers=4,
        num_decoder_layers=4, dim_feedforward=1024, dropout=0.1,
        encoder_embed_dim=64, **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.stride = stride
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.encoder_embed_dim = encoder_embed_dim


class WhipstrTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer using your model.json vocabulary.
    
    Token ID scheme:
        0 = PAD / BOS / UNK  (shared)
        1..N = real characters
        N+1 = EOS
    """
    
    pad_token = "<pad>"
    eos_token = "<eos>"
    
    def __init__(self, vocab_file, **kwargs):
        with open(vocab_file) as f:
            vocab_list = json.load(f)["Vocab"]
        
        self.char_to_idx = {c: i + 1 for i, c in enumerate(vocab_list)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(vocab_list)}
        self.char_to_idx[self.pad_token] = 0
        self.idx_to_char[0] = self.pad_token
        self.char_to_idx[self.eos_token] = len(vocab_list) + 1
        self.idx_to_char[len(vocab_list) + 1] = self.eos_token
        
        super().__init__(pad_token=self.pad_token, eos_token=self.eos_token, **kwargs)
    
    @property
    def vocab_size(self):
        return len(self.char_to_idx)
    
    def _tokenize(self, text):
        return list(text)
    
    def _convert_token_to_id(self, token):
        return self.char_to_idx.get(token, 0)
    
    def _convert_id_to_token(self, index):
        return self.idx_to_char.get(index, self.pad_token)
    
    def get_vocab(self):
        return self.char_to_idx.copy()
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = os.path.join(save_directory, "model.json")
        with open(path, "w") as f:
            vocab_list = list(self.char_to_idx.keys())
            for special in [self.pad_token, self.eos_token]:
                if special in vocab_list:
                    vocab_list.remove(special)
            json.dump({"Vocab": vocab_list}, f, indent=2)
        return (path,)


class WhipstrFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_features"]
    
    def __init__(self, sampling_rate=48000, window_size=1280, resolut=4096, num_freqs=None, 
                 y_reverse=True, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.resolut = resolut
        self.y_reverse = y_reverse
        
        if num_freqs is not None:
            self.num_freqs = num_freqs
        else:
            self._set_num_freqs_from_sampling_rate()
    
    def _set_num_freqs_from_sampling_rate(self):
        """Set num_freqs based on sampling rate (matches Phase behavior)."""
        if self.sampling_rate in [8000, 16000, 24000, 32000, 48000]:
            self.num_freqs = 768
        elif self.sampling_rate in [11025, 22050, 44100]:
            self.num_freqs = 836
        else:
            raise ValueError(
                f"Unsupported sample rate: {self.sampling_rate}. "
                f"Supported rates are: 8000, 16000, 24000, 32000, 48000, 11025, 22050, 44100"
            )
    
    def _zero_pad(self):
        """Return zero_pad parameter for upsampling."""
        if self.num_freqs == 768:  # 48kHz family
            return 0
        else:  # 44.1kHz family
            return 1
    
    def _zero_shift(self):
        """Return zero_shift parameter for upsampling."""
        if self.num_freqs == 768:  # 48kHz family
            if self.sampling_rate == 48000:
                return 0
            elif self.sampling_rate == 32000:
                return 1
            elif self.sampling_rate == 24000:
                return 1
            elif self.sampling_rate == 16000:
                return 2
            elif self.sampling_rate == 8000:
                return 5
        else:  # 44.1kHz family
            if self.sampling_rate == 44100:
                return 0
            elif self.sampling_rate == 22050:
                return 1
            elif self.sampling_rate == 11025:
                return 3
        return 0
    
    def _load_audio_file(self, audio_path):
        import soundfile as sf
        
        audio, sr = sf.read(audio_path, dtype='float64')
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        return audio
    
    def _load_audio_file_with_sr(self, audio_path):
        import soundfile as sf
        
        audio, sr = sf.read(audio_path, dtype='float64')
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        return audio, sr
    
    def __call__(self, audio, sampling_rate=None, return_tensors="pt"):
        if isinstance(audio, str):
            audio, input_sr = self._load_audio_file_with_sr(audio)
        else:
            input_sr = sampling_rate
        
        # Upsample to one of two rates: 48000 (768 freqs) or 44100 (836 freqs)
        if self.num_freqs == 768:
            audio = self._upsample_to_48k(audio, input_sr)
        else:  # num_freqs == 836
            audio = self._upsample_to_44k(audio, input_sr)
        
        spectrogram = self._audio_to_spectrogram(audio)
        features = self._preprocess(spectrogram)
        
        if return_tensors == "pt":
            return {"input_features": features}
        else:
            raise ValueError(f"Unsupported return_tensors: {return_tensors}")
    
    def _upsample_to_48k(self, audio, input_sr):
        """Upsample audio for 48kHz family to 48000 Hz."""
        if input_sr == 48000:
            return audio
        elif input_sr == 32000:
            zero_pad, zero_shift = 2, 1
        elif input_sr == 24000:
            zero_pad, zero_shift = 1, 1
        elif input_sr == 16000:
            zero_pad, zero_shift = 1, 2
        elif input_sr == 8000:
            zero_pad, zero_shift = 1, 5
        else:
            return audio
        
        if zero_pad == 0:
            return audio
        
        num_groups = (len(audio) + zero_pad - 1) // zero_pad
        output_len = len(audio) + num_groups * zero_shift
        output = np.zeros(output_len, dtype=audio.dtype)
        
        out_idx = 0
        boost = 1 + zero_shift
        for i in range(len(audio)):
            output[out_idx] = audio[i] * boost
            out_idx += 1
            if (i + 1) % zero_pad == 0:
                out_idx += zero_shift
        
        return output
    
    def _upsample_to_44k(self, audio, input_sr):
        """Upsample audio for 44.1kHz family to 44100 Hz."""
        if input_sr == 44100:
            return audio
        elif input_sr == 22050:
            zero_pad, zero_shift = 1, 1
        elif input_sr == 11025:
            zero_pad, zero_shift = 1, 3
        else:
            return audio
        
        if zero_pad == 0:
            return audio
        
        num_groups = (len(audio) + zero_pad - 1) // zero_pad
        output_len = len(audio) + num_groups * zero_shift
        output = np.zeros(output_len, dtype=audio.dtype)
        
        out_idx = 0
        boost = 1 + zero_shift
        for i in range(len(audio)):
            output[out_idx] = audio[i] * boost
            out_idx += 1
            if (i + 1) % zero_pad == 0:
                out_idx += zero_shift
        
        return output
    
    def _audio_to_spectrogram(self, audio_buffer):
        """Convert audio to phase-preserving spectrogram (matches Phase.to_phase)."""
        padded_audio = self._pad(audio_buffer)
        
        hop_size = self.window_size
        frame_len = self.resolut
        num_frames = int((len(padded_audio) - frame_len) / hop_size) + 1
        hann_window = np.hanning(frame_len)
        
        indices = np.arange(frame_len)[None, :] + np.arange(num_frames)[:, None] * hop_size
        frames = padded_audio[indices] * hann_window
        stft_result = np.fft.fft(frames, axis=1).T
        
        num_bins = self.resolut // 2
        j = np.arange(num_bins)
        v0 = stft_result[j + 1, :]
        v1 = stft_result[self.resolut - j - 1, :]
        
        realn1 = np.imag(v0)
        realm0 = np.real(v1)
        
        phase_repr = np.stack([realn1, realm0], axis=2)
        phase_repr = phase_repr.transpose(1, 0, 2).reshape(-1, 2).astype(np.float64)
        
        return self._shrink(phase_repr, num_frames)
    
    def _pad(self, audio_buffer):
        current_len = len(audio_buffer)
        min_target_size = 15 * self.window_size
        pad_len = 0
        
        if current_len >= min_target_size:
            remainder = (current_len - min_target_size) % self.window_size
            if remainder != 0:
                pad_len = self.window_size - remainder - 1
        else:
            pad_len = min_target_size - current_len - 1
        
        if pad_len > 0:
            return np.pad(audio_buffer, (0, pad_len), mode='constant', constant_values=0)
        return audio_buffer
    
    def _shrink(self, spectrogram, time_frames):
        original_bins = self.resolut // 2
        return spectrogram.reshape(time_frames, original_bins, 2)[:, :self.num_freqs, :].reshape(-1, 2)
    
    def _preprocess(self, spectrogram):
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.from_numpy(spectrogram).float()
        
        time_frames = spectrogram.shape[0] // self.num_freqs
        
        spectrogram = spectrogram.reshape(time_frames, self.num_freqs, 2)
        spectrogram = spectrogram.permute(2, 1, 0)
        
        if spectrogram.shape[1] < 836:
            padding = torch.zeros(spectrogram.shape[0], 836 - spectrogram.shape[1], spectrogram.shape[2])
            spectrogram = torch.cat([spectrogram, padding], dim=1)
        elif spectrogram.shape[1] > 836:
            spectrogram = spectrogram[:, :836, :]
        
        return spectrogram.unsqueeze(0)


class WhipstrForConditionalGeneration(PreTrainedModel):
    config_class = WhipstrConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        self.encoder = WhipstrEncoder(
            stride=config.stride, window_size=config.window_size,
            output_values=config.encoder_embed_dim,
        )
        self.transformer = WhipstrTransformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            vocab_size=config.vocab_size,
            input_values=config.encoder_embed_dim,
        )
        
        self.post_init()
    
    def forward(self, input_features, labels=None):
        encoder_tokens = self.encoder(input_features)
        outputs = self.transformer(encoder_tokens, labels) if labels is not None else encoder_tokens
        return {"logits": outputs}
    
    def generate(self, input_features, max_length=500, start_token=0, eos_token=None, **kwargs):
        if eos_token is None:
            eos_token = self.config.vocab_size - 1
        
        encoder_tokens = self.encoder(input_features)
        return self.transformer.generate(encoder_tokens, max_length=max_length, start_token=start_token, eos_token=eos_token)


def convert_to_hf(checkpoint_path, model_json_path, output_dir, vocab_size=None, variant=None):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(model_json_path) as f:
        vocab_data = json.load(f)
        vocab_list = vocab_data["Vocab"]
        vocab_size = vocab_size if vocab_size else (len(vocab_list) + 2)
    
    if variant:
        from .whipstr_variants import get_hf_config
        config = get_hf_config(variant, vocab_size=vocab_size)
    else:
        config = WhipstrConfig(
            vocab_size=vocab_size, stride=1, window_size=11,
            d_model=256, nhead=8, num_encoder_layers=4,
            num_decoder_layers=4, dim_feedforward=1024, dropout=0.1
        )
    
    model = WhipstrForConditionalGeneration(config)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.transformer.load_state_dict(checkpoint["transformer_state_dict"])
    
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    tokenizer = WhipstrTokenizer(model_json_path)
    tokenizer.save_pretrained(output_dir)
    
    feature_extractor = WhipstrFeatureExtractor()
    feature_extractor.save_pretrained(output_dir)
    
    print(f"HF model saved to {output_dir}")
    print("  - pytorch_model.bin")
    print("  - config.json")
    print("  - model.json")
    print("  - preprocessor_config.json")


if __name__ == "__main__":
    import argparse
    from .whipstr_variants import list_variants
    parser = argparse.ArgumentParser(description="Convert Whipstr STT checkpoint to Hugging Face format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--model-json", type=str, required=True, help="Path to model.json (vocab file)")
    parser.add_argument("--output-dir", type=str, default="./hf_whipstr", help="Output directory for HF model")
    parser.add_argument("--variant", type=str, default=None,
                        choices=list_variants(),
                        help="Model variant (if omitted, uses hardcoded base config)")
    
    args = parser.parse_args()
    
    convert_to_hf(args.checkpoint, args.model_json, args.output_dir, variant=args.variant)
    
    print("\nTo upload to Hugging Face Hub:")
    print("  huggingface-cli login")
    print("  huggingface-cli upload ./hf_whipstr neuralang/en-whipstr-base-48khz-libritts-r")
