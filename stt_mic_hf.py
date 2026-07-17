"""
Simple microphone-based inference script for Whipstr STT with Hugging Face Hub support.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from transformers import AutoModel, AutoConfig

from whipstr.hf_integration import WhipstrConfig, WhipstrForConditionalGeneration, WhipstrTokenizer, WhipstrFeatureExtractor
AutoConfig.register("whipstr", WhipstrConfig)
AutoModel.register(WhipstrConfig, WhipstrForConditionalGeneration)


def infer_from_mic(model_path, duration, sampling_rate, device=None, save_audio=None):
    """Record from microphone and run inference."""

    import sounddevice as sd
    import soundfile as sf

    print(f"Recording for {duration} seconds...", file=sys.stderr)
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate,
                   device=device, channels=1, dtype='float64')
    sd.wait()
    audio = audio.flatten()
    print("Recording finished.", file=sys.stderr)

    if save_audio:
        sf.write(save_audio, audio, sampling_rate)
        print(f"Audio saved to {save_audio}", file=sys.stderr)

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    config = model.config
    model.eval()

    from transformers.utils import cached_file
    vocab_file = cached_file(model_path, 'model.json', trust_remote_code=True)
    tokenizer = WhipstrTokenizer(vocab_file)

    feature_extractor = WhipstrFeatureExtractor(sampling_rate=sampling_rate)
    extracted = feature_extractor(audio, sampling_rate=sampling_rate)
    input_features = extracted["input_features"]

    with torch.no_grad():
        pad_id = 0
        eos_id = config.vocab_size - 1
        actual_frames = input_features.shape[-1]
        estimated_chars = max(actual_frames // 4, 100)

        encoder_tokens = model.encoder(input_features)

        predictions = model.transformer.generate(
            encoder_tokens,
            max_length=estimated_chars + 100,
            start_token=pad_id,
            eos_token=eos_id,
        )

        predicted_indices = predictions[0].cpu().tolist()

        if eos_id in predicted_indices:
            predicted_indices = predicted_indices[:predicted_indices.index(eos_id)]

        predicted_text = ''.join(
            tokenizer.idx_to_char.get(idx, '?')
            for idx in predicted_indices
            if 0 < idx < eos_id
        )

    return predicted_text


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Whipstr STT Microphone Inference')
    parser.add_argument('--duration', type=float, default=5,
                        help='Recording duration in seconds (default: 5)')
    parser.add_argument('--sampling-rate', type=int, default=48000,
                        help='Sample rate for microphone capture (default: 48000)')
    parser.add_argument('--device', type=int, default=None,
                        help='Input device index (default: system default)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--model', type=str, default='./hf_whipstr',
                        help='Hugging Face model path or ID')
    parser.add_argument('--save-audio', type=str, default=None,
                        help='Save recorded audio to this WAV file')

    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if args.sampling_rate not in (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000):
        print(f"Warning: unsupported sampling rate {args.sampling_rate}. "
              f"Supported: 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000",
              file=sys.stderr)
        sys.exit(1)

    transcription = infer_from_mic(
        args.model, args.duration, args.sampling_rate,
        device=args.device, save_audio=args.save_audio
    )
    print(f"Transcription: {transcription}")


if __name__ == '__main__':
    main()
