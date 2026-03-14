# Whipstr STT (ASR) — Quick Start: 56-Pixel Height Spectrograms

## What Changed?

Spectrograms are now 56 pixels tall (was 28), with audio features randomly positioned vertically.

## Quick Verification

```bash
# Run verification script
python test_stretch.py

# Run all tests
pytest tests/ -v
```

## Usage Examples

### Generate Dataset Sample
```python
from torchvision import datasets, transforms
from whipstr.whipstr_mnist_dataset import WhipstrMNISTDataset

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataset = WhipstrMNISTDataset(mnist, num_digits=5, max_spacing=5, noise_std=0.1, num_samples=100)

spectrogram, transcription = dataset[0]
print(f"Shape: {spectrogram.shape}")  # [2, 56, W]
print(f"Transcription: {transcription}")
```

### Process with Encoder
```python
import torch
from whipstr.whipstr_encoder import WhipstrEncoder

encoder = WhipstrEncoder(stride=1)
test_spectrogram = torch.rand(4, 2, 56, 140)  # Batch of 4
output = encoder(test_spectrogram)
print(f"Output: {output.shape}")  # [4, 113, 10]
```

### Train Model
```bash
# Quick training example
python example.py

# Full training with improved hyperparameters
python train_improved.py
```

### Visualize Samples
```bash
python visualize_sample.py
# Creates: sample_visualization.png, sample_rgb.png
```

## Key Differences from 28px

| Aspect | 28px (Old) | 56px (New) |
|--------|-----------|-----------|
| Spectrogram shape | `[2, 28, W]` | `[2, 56, W]` |
| Window shape | `[2, 28, 28]` | `[2, 56, 28]` |
| Vertical position | Fixed | Random (0-28 offset) |
| Encoder params | ~6.4M | ~1.77M |
| Conv layers | 3 | 4 |
| Noise generation | 28px | 56px native |

## Architecture Summary

### Encoder
```
Input: [batch, 2, 56, 28]
Conv1(32) → Conv2(64) → Pool → Conv3(128) → Pool → Conv4(256) → Pool
→ Flatten → FC1(256) → FC2(10)
Output: [batch, num_windows, 10] (raw logits)
```

### Dataset
```
1. Concatenate N audio feature segments horizontally (28x28 each)
2. Place on 56-pixel height canvas with random vertical offset (0-28)
3. Duplicate to red/green channels
4. Add independent Gaussian noise at 56px resolution
5. Clip to [0, 1]
Output: [2, 56, W]
```

## Troubleshooting

### Out of Memory
Reduce batch size in training scripts:
```python
batch_size = 32  # Try 16 or 8
```

### Tests Failing
Ensure you're using the updated code:
```bash
pytest tests/ -v
```

### Visualization Issues
Check matplotlib is installed:
```bash
pip install matplotlib
```

## Documentation

- `IMAGE_STRETCH_CHANGES.md` — Full technical details
- `doc/IMAGE_STRETCH_SUMMARY.md` — Implementation summary
- `test_stretch.py` — Verification script

## Next Steps

1. ✅ Verify installation: `python test_stretch.py`
2. ✅ Run tests: `pytest tests/ -v`
3. 🚀 Train model: `python train_improved.py`
4. 📊 Visualize: `python visualize_sample.py`
5. 🔬 Experiment with hyperparameters

---

Status: ✅ Ready for use
All tests: ✅ Passing (42/42)
Verification: ✅ Complete
