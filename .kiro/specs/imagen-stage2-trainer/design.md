# Design Document — Imagen Stage 2 Trainer

## Overview

Stage 2 trains an **asymmetric 2D U-Net** (`ImageGenerator`) as the encoder half of an autoencoder whose decoder is the frozen Stage 1 `WhipstrEncoder`. The generator takes a noisy spectrogram window `(batch, 2, 11, 836)` plus a 64-float conditioning vector and predicts the noise that was added (diffusion-style). An auxiliary loss passes the denoised prediction through the frozen CNN to compare token outputs against those of the clean input, anchoring the generator to acoustically meaningful reconstructions.

All new code lives in `imagen/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE 2 AUTOENCODER                      │
│                                                                 │
│  x_t (2,11,836)  ──►  ImageGenerator (trainable)  ──►  noise_pred (2,11,836)  │
│                              │                                  │
│                    cond (64,) via MLP                           │
│                                                                 │
│  x_denoised = x_t - noise_pred                                  │
│       │                                                         │
│       ▼                                                         │
│  WhipstrEncoder (frozen)  ──►  tokens_pred                      │
│                                                                 │
│  x_clean ──► WhipstrEncoder (frozen) ──► tokens_clean           │
│                                                                 │
│  Loss = MSE(noise_pred, noise) + λ * MSE(tokens_pred, tokens_clean) │
└─────────────────────────────────────────────────────────────────┘
```

### Diffusion training loop (single step, simplified)

```
noise ~ N(0, 1),  t ~ Uniform(0, 1)
x_t   = x_clean + t * noise
noise_pred = ImageGenerator(x_t, cond)
loss_noise = MSE(noise_pred, noise)
x_denoised = x_t - noise_pred
loss_recon  = MSE(encoder(x_denoised), encoder(x_clean))   # frozen encoder
loss = loss_noise + lambda_recon * loss_recon
```

---

## Components and Interfaces

### `imagen/image_generator.py` — `ImageGenerator`

The asymmetric 2D U-Net. Input `(batch, 2, 11, 836)`, output `(batch, 2, 11, 836)`.

```
ImageGenerator(
    in_channels=2,
    base_channels=64,
    cond_dim=64,
    cond_hidden=128
)
```

### `imagen/conditioning_mlp.py` — `ConditioningMLP`

Projects the 64-float conditioning vector to a spatial bias added at the bottleneck.

```
ConditioningMLP(in_dim=64, hidden_dim=128, out_dim=bottleneck_channels)
```

### `imagen/spectrogram_window_dataset.py` — `SpectrogramWindowDataset`

Wraps `WhipstrTSVSpeechDataset`, slices each full spectrogram into non-overlapping `(2, 11, 836)` windows, and produces a 64-float conditioning vector per window.

### `imagen/imagen_train.py` — `ImagenTrainer`

Orchestrates the diffusion training loop: noise sampling, forward pass, loss computation, backward pass, checkpointing, and logging.

---

## Data Models

### Window tensor
```
shape : (2, 11, 836)   # channels=2, time=11, freq=836
dtype : float32
range : unbounded (raw phase spectrogram values)
```

### Conditioning vector
```
shape : (64,)
dtype : float32
content: [window_position_normalized (1), mean_ch0 (1), std_ch0 (1),
          mean_ch1 (1), std_ch1 (1), zero-padded to 64]
```

### Checkpoint file (`.pt`)
```python
{
  'epoch': int,
  'generator_state_dict': OrderedDict,
  'optimizer_state_dict': OrderedDict,
  'loss_noise': float,
  'loss_recon': float,
  'config': dict   # ImageGenerator constructor kwargs
}
```

---

## U-Net Architecture Detail

### Encoder path (downsampling)

| Layer | Op | In shape | Out shape |
|---|---|---|---|
| time_down | Conv2d(2→64, k=(11,1), s=(11,1)) | (B,2,11,836) | (B,64,1,836) |
| freq_down1 | Conv2d(64→128, k=(1,3), s=(1,2), p=(0,1)) | (B,64,1,836) | (B,128,1,418) |
| freq_down2 | Conv2d(128→256, k=(1,3), s=(1,2), p=(0,1)) | (B,128,1,418) | (B,256,1,209) |
| freq_down3 | Conv2d(256→256, k=(1,3), s=(1,2), p=(0,1)) | (B,256,1,209) | (B,256,1,105) |

### Bottleneck

| Layer | Op | Shape |
|---|---|---|
| res_block | Conv2d(256→256, k=(1,3), p=(0,1)) × 2 + residual | (B,256,1,105) |
| cond_add | add projected cond vector (broadcast over freq) | (B,256,1,105) |

### Decoder path (upsampling)

| Layer | Op | In shape | Out shape |
|---|---|---|---|
| freq_up3 | ConvTranspose2d(512→256, k=(1,4), s=(1,2), p=(0,1)) | (B,512,1,105) | (B,256,1,209) |
| freq_up2 | ConvTranspose2d(512→128, k=(1,4), s=(1,2), p=(0,1)) | (B,512,1,209) | (B,128,1,418) |
| freq_up1 | ConvTranspose2d(256→64, k=(1,4), s=(1,2), p=(0,1)) | (B,256,1,418) | (B,64,1,836) |
| time_up | ConvTranspose2d(128→2, k=(11,1), s=(11,1)) | (B,128,1,836) | (B,2,11,836) |

Skip connections concatenate encoder feature maps to decoder inputs (hence doubled channel counts at decoder inputs).

> Note: `ConvTranspose2d` with `k=(1,4), s=(1,2), p=(0,1)` is the standard "double" transpose conv that exactly inverts `Conv2d(k=(1,3), s=(1,2), p=(0,1))` for even-sized inputs. Odd sizes (209) require an `output_padding=(0,1)` argument.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Output shape identity

*For any* batch of spectrogram windows of shape `(B, 2, 11, 836)` and any conditioning vector of shape `(B, 64)`, the `ImageGenerator` output SHALL have the same shape `(B, 2, 11, 836)`.

**Validates: Requirements 1.1**

---

### Property 2: Time axis collapse and restore

*For any* valid input, the intermediate representation after `time_down` SHALL have time dimension 1, and the final output SHALL have time dimension 11.

**Validates: Requirements 1.2, 1.5**

---

### Property 3: Frequency bottleneck size

*For any* valid input, the bottleneck feature map SHALL have a frequency dimension ≤ 105.

**Validates: Requirements 1.3**

---

### Property 4: Conditioning vector influence

*For any* two distinct conditioning vectors `c1 ≠ c2` applied to the same noisy input, the `ImageGenerator` SHALL produce different outputs.

**Validates: Requirements 2.1, 2.2**

---

### Property 5: Frozen encoder invariance

*For any* input passed through the frozen `WhipstrEncoder` before and after Stage 2 training begins, the encoder SHALL produce identical outputs (weights unchanged).

**Validates: Requirements 3.2**

---

### Property 6: Noise prediction finite values

*For any* finite input window and conditioning vector, the `ImageGenerator` output SHALL contain only finite values (no NaN or Inf).

**Validates: Requirements 1.1, 4.2**

---

### Property 7: Checkpoint round-trip

*For any* `ImageGenerator` model, saving with `torch.save` and reloading with `torch.load` SHALL produce a model whose `state_dict` is identical to the original.

**Validates: Requirements 7.1, 7.3**

---

### Property 8: Optimizer state round-trip

*For any* trainer checkpoint, saving and reloading SHALL restore the optimizer state dict and epoch counter to their exact saved values.

**Validates: Requirements 7.2**

---

### Property 9: Dataset window shape

*For any* spectrogram of shape `(2, 836, W)` where `W ≥ 11`, the `SpectrogramWindowDataset` SHALL yield windows of shape `(2, 11, 836)` and conditioning vectors of shape `(64,)`.

**Validates: Requirements 6.2, 6.4**

---

### Property 10: Dataset window count

*For any* spectrogram of width `W`, the number of windows yielded SHALL equal `W // 11`.

**Validates: Requirements 6.2, 6.3**

---

## Error Handling

| Condition | Behaviour |
|---|---|
| Checkpoint file not found | `FileNotFoundError` with descriptive message |
| Stage 1 weights path missing | `FileNotFoundError` raised at trainer init |
| Input shape mismatch | `ValueError` with expected vs actual shape |
| NaN in input | `ValueError` before forward pass |
| CUDA OOM | Re-raise `RuntimeError` with shape info and suggestion to reduce batch size |
| Width < 11 in dataset | `ValueError` — cannot extract even one window |

---

## Testing Strategy

### Property-based testing library

**`hypothesis`** (already used in the project, see `tests/test_whipstr_encoder.py`). Each property test runs a minimum of **100 iterations** (`@settings(max_examples=100, deadline=None)`).

### Annotation format

Every property-based test MUST be tagged with:
```python
# Feature: imagen-stage2-trainer, Property N: <property text>
# Validates: Requirements X.Y
```

### Unit tests

- Shape assertions for each U-Net layer in isolation
- Frozen encoder: verify `requires_grad=False` on all parameters after freezing
- Dataset: correct window count and shape for synthetic spectrograms
- Checkpoint save/load: exact state dict equality
- Conditioning MLP: output shape `(B, bottleneck_channels, 1, 1)`

### Property-based tests (one test per property)

| Test | Property | Hypothesis strategy |
|---|---|---|
| `test_property_1_output_shape` | P1 | `st.integers` for batch/width |
| `test_property_2_time_axis` | P2 | random batch sizes |
| `test_property_3_freq_bottleneck` | P3 | random batch sizes |
| `test_property_4_conditioning_influence` | P4 | random cond vectors |
| `test_property_5_frozen_encoder` | P5 | random inputs |
| `test_property_6_finite_output` | P6 | random finite inputs |
| `test_property_7_checkpoint_roundtrip` | P7 | random model weights |
| `test_property_8_optimizer_roundtrip` | P8 | random optimizer states |
| `test_property_9_dataset_window_shape` | P9 | `st.integers` for W |
| `test_property_10_dataset_window_count` | P10 | `st.integers` for W |
