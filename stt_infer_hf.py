"""
Simple inference script for Whipstr STT with Hugging Face Hub support.
"""

import sys
sys.path.insert(0, '.')

import torch
from transformers import AutoModel, AutoConfig

from whipstr.hf_integration import (
    WhipstrConfig, 
    WhipstrForConditionalGeneration,
    WhipstrTokenizer,
    WhipstrFeatureExtractor
)
AutoConfig.register("whipstr", WhipstrConfig)
AutoModel.register(WhipstrConfig, WhipstrForConditionalGeneration)


def infer(audio_path, model_path):
    """Run inference on audio file."""
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    config = model.config
    model.eval()
    
    from transformers.utils import cached_file
    vocab_file = cached_file(model_path, 'model.json', trust_remote_code=True)
    tokenizer = WhipstrTokenizer(vocab_file)
    
    # Load feature extractor and process audio
    # Determine sampling rate from file to set correct num_freqs (matches Phase behavior)
    import soundfile as sf
    _, file_sr = sf.read(audio_path, dtype='float64')
    if isinstance(file_sr, tuple):
        file_sr = file_sr[1]
    
    feature_extractor = WhipstrFeatureExtractor(sampling_rate=file_sr)
    extracted = feature_extractor(audio_path, sampling_rate=None)
    input_features = extracted["input_features"]
    
    with torch.no_grad():
        start_token_idx = config.vocab_size - 1
        
        # Estimate max_length based on audio width (rough estimate: 1 char per 10-20 frames)
        # encoder produces ~width/window_size tokens, so max tokens ~ width/window_size
        # window_size=11, so roughly width/11 tokens
        actual_frames = input_features.shape[-1]
        estimated_chars = max(actual_frames // 4, 100)  # at least 50 chars
        
        encoder_tokens = model.encoder(input_features)
        
        predictions = model.transformer.generate(
            encoder_tokens,
            max_length=estimated_chars + 100,  # add buffer
            start_token=start_token_idx
        )
        
        predicted_indices = predictions[0].cpu().tolist()
        
        predicted_text = ''.join(
            tokenizer.idx_to_char.get(idx, '?')
            for idx in predicted_indices
            if 0 < idx < config.vocab_size
        )
    
    return predicted_text


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Whipstr STT Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file (WAV/FLAC)')
    parser.add_argument('--model', type=str, default='./hf_whipstr',
                        help='Hugging Face model path or ID')
    
    args = parser.parse_args()
    
    transcription = infer(args.audio, args.model)
    print(f"Transcription: {transcription}")


if __name__ == '__main__':
    main()
