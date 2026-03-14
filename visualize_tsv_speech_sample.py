import torch
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
import matplotlib.pyplot as plt
import numpy as np

# Create TSV Speech dataset
dataset = WhipstrTSVSpeechDataset(
    tsv_path='data/TSV_SPEECH/speech.tsv',
    noise_std=0.1
)

print(f"Dataset size: {len(dataset)} samples\n")

# Visualize all samples
for idx in range(len(dataset)):
    image, transcription = dataset[idx]
    
    print(f"Sample {idx}:")
    print(f"  Transcription: '{transcription}'")
    print(f"  Image shape: {image.shape}")
    print(f"  Image value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Normalize for visualization: map zero to 0.5
    red_channel = image[0].numpy()
    green_channel = image[1].numpy()
    
    # Find max absolute value for symmetric scaling
    red_max_abs = max(abs(red_channel.min()), abs(red_channel.max()))
    green_max_abs = max(abs(green_channel.min()), abs(green_channel.max()))
    
    # Normalize: map [-max_abs, +max_abs] to [0, 1] with 0 -> 0.5
    if red_max_abs > 0:
        red_norm = (red_channel / red_max_abs) * 0.5 + 0.5
    else:
        red_norm = np.full_like(red_channel, 0.5)
    
    if green_max_abs > 0:
        green_norm = (green_channel / green_max_abs) * 0.5 + 0.5
    else:
        green_norm = np.full_like(green_channel, 0.5)
    
    # Clamp to [0, 1]
    red_norm = np.clip(red_norm, 0.0, 1.0)
    green_norm = np.clip(green_norm, 0.0, 1.0)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(1, 8))
    
    # Show red channel
    axes[0].imshow(red_norm, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f'Red Channel - Transcription: "{transcription}"', fontsize=14)
    axes[0].set_ylabel('Frequency (836 bins)', fontsize=10)
    axes[0].set_xlabel('Time', fontsize=10)
    
    # Show green channel
    axes[1].imshow(green_norm, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f'Green Channel - Transcription: "{transcription}"', fontsize=14)
    axes[1].set_ylabel('Frequency (836 bins)', fontsize=10)
    axes[1].set_xlabel('Time', fontsize=10)
    
    plt.tight_layout()
    filename = f'tsv_speech_sample_{idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    
    # Also create a combined RGB visualization
    fig, ax = plt.subplots(1, 1, figsize=(1, 6))
    
    # Create RGB image (red and green channels, blue is zero)
    height, width = image.shape[1], image.shape[2]
    rgb_image = np.zeros((height, width, 3))
    rgb_image[:, :, 0] = red_norm  # Red channel (normalized)
    rgb_image[:, :, 1] = green_norm  # Green channel (normalized)
    # Blue channel stays 0
    
    ax.imshow(rgb_image, aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Combined RGB View - Transcription: "{transcription}"', fontsize=14)
    ax.set_ylabel(f'Frequency ({height} bins)', fontsize=10)
    ax.set_xlabel('Time', fontsize=10)
    
    plt.tight_layout()
    rgb_filename = f'tsv_speech_sample_{idx}_rgb.png'
    plt.savefig(rgb_filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {rgb_filename}\n")
    
    plt.close('all')

print("All visualizations complete!")
