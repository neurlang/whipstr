"""
Quick test script for Whipstr STT (ASR) WhipstrTSVSpeechDataset
"""
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset

# Create dataset
dataset = WhipstrTSVSpeechDataset('data/TSV_SPEECH/speech.tsv', noise_std=0.1)

print(f"Dataset size: {len(dataset)}")

# Test loading first sample
image_tensor, transcription = dataset[0]
print(f"\nFirst sample:")
print(f"  Image shape: {image_tensor.shape}")
print(f"  Transcription: '{transcription}'")
print(f"  Image value range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

# Test all samples
print(f"\nAll samples:")
for i in range(len(dataset)):
    img, text = dataset[i]
    print(f"  [{i}] Shape: {img.shape}, Text: '{text}'")
