# Implementation Plan

- [x] 1. Set up `imagen/` package structure and shared utilities
  - Create `imagen/__init__.py`
  - Add input validation helpers (shape checks, NaN/Inf guards) reusable across modules
  - _Requirements: 1.1, 4.2_

- [x] 2. Implement `ConditioningMLP`
- [x] 2.1 Create `imagen/conditioning_mlp.py` with `ConditioningMLP(in_dim=64, hidden_dim=128, out_dim)`
  - Two linear layers with ReLU between them
  - Output shape `(B, out_dim)` suitable for broadcast-adding to bottleneck
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2.2 Write property test for conditioning MLP output shape (Property 4)
  - **Feature: imagen-stage2-trainer, Property 4: Conditioning vector influence**
  - **Validates: Requirements 2.1, 2.2**

- [x] 3. Implement `ImageGenerator` U-Net
- [x] 3.1 Create `imagen/image_generator.py` — encoder path
  - `time_down`: `Conv2d(2→64, k=(11,1), s=(11,1))`
  - `freq_down1/2/3`: stride-2 convs along frequency axis (836→418→209→105)
  - Store skip connection tensors at each freq level
  - _Requirements: 1.2, 1.3_

- [x] 3.2 Write property test for time axis collapse (Property 2)
  - **Feature: imagen-stage2-trainer, Property 2: Time axis collapse and restore**
  - **Validates: Requirements 1.2, 1.5**

- [x] 3.3 Write property test for frequency bottleneck size (Property 3)
  - **Feature: imagen-stage2-trainer, Property 3: Frequency bottleneck size**
  - **Validates: Requirements 1.3**

- [x] 3.4 Add bottleneck residual block and conditioning injection
  - Two `Conv2d(256→256, k=(1,3), p=(0,1))` with residual connection
  - Call `ConditioningMLP`, reshape output to `(B, C, 1, 1)`, add to bottleneck
  - _Requirements: 2.2, 4.1_

- [x] 3.5 Add decoder path (frequency upsampling + time upsampling)
  - `freq_up3/2/1`: `ConvTranspose2d` with skip concatenation at each level
  - `time_up`: `ConvTranspose2d(128→2, k=(11,1), s=(11,1))`
  - _Requirements: 1.4, 1.5, 1.6_

- [x] 3.6 Add `get_config` method returning constructor hyperparameters dict
  - _Requirements: 7.3_

- [x] 3.7 Write property test for output shape identity (Property 1)
  - **Feature: imagen-stage2-trainer, Property 1: Output shape identity**
  - **Validates: Requirements 1.1**

- [x] 3.8 Write property test for finite output values (Property 6)
  - **Feature: imagen-stage2-trainer, Property 6: Noise prediction finite values**
  - **Validates: Requirements 1.1, 4.2**

- [x] 3.9 Write property test for conditioning influence (Property 4)
  - **Feature: imagen-stage2-trainer, Property 4: Conditioning vector influence**
  - **Validates: Requirements 2.1, 2.2**

- [x] 3.10 Write property test for checkpoint round-trip (Property 7)
  - **Feature: imagen-stage2-trainer, Property 7: Checkpoint round-trip**
  - **Validates: Requirements 7.1, 7.3**

- [x] 4. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement `SpectrogramWindowDataset`
- [x] 5.1 Create `imagen/spectrogram_window_dataset.py`
  - Accept a list of `(2, 836, W)` tensors or a TSV path (reuse `WhipstrTSVSpeechDataset`)
  - Slice each spectrogram into non-overlapping `(2, 11, 836)` windows (transpose time/freq axes)
  - Discard remainder when `W % 11 != 0`
  - Build 64-float conditioning vector: `[position_norm, mean_ch0, std_ch0, mean_ch1, std_ch1, zeros...]`
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 5.2 Write property test for dataset window shape (Property 9)
  - **Feature: imagen-stage2-trainer, Property 9: Dataset window shape**
  - **Validates: Requirements 6.2, 6.4**

- [x] 5.3 Write property test for dataset window count (Property 10)
  - **Feature: imagen-stage2-trainer, Property 10: Dataset window count**
  - **Validates: Requirements 6.2, 6.3**

- [ ] 6. Implement `ImagenTrainer`
- [ ] 6.1 Create `imagen/imagen_train.py` — trainer class and noise injection
  - Load `WhipstrEncoder` from checkpoint path, freeze all parameters (`requires_grad=False`)
  - Implement `add_noise(x_clean, t)` → `x_t = x_clean + t * noise`, return `(x_t, noise)`
  - _Requirements: 3.1, 3.2, 4.1_

- [ ] 6.2 Write property test for frozen encoder invariance (Property 5)
  - **Feature: imagen-stage2-trainer, Property 5: Frozen encoder invariance**
  - **Validates: Requirements 3.2**

- [ ] 6.3 Implement loss computation
  - `loss_noise = MSE(noise_pred, noise)`
  - `loss_recon = MSE(encoder(x_denoised), encoder(x_clean))` with frozen encoder
  - `total_loss = loss_noise + lambda_recon * loss_recon`
  - _Requirements: 4.2, 4.3, 4.4_

- [ ] 6.4 Implement training loop with logging and checkpointing
  - Epoch loop over `SpectrogramWindowDataset` via `DataLoader`
  - Log `loss_noise` and `loss_recon` per epoch to stdout
  - Save checkpoint every N epochs to configurable output directory
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 6.5 Implement checkpoint resume logic
  - Load checkpoint, restore `generator_state_dict`, `optimizer_state_dict`, and `epoch`
  - _Requirements: 5.4_

- [ ] 6.6 Write property test for optimizer state round-trip (Property 8)
  - **Feature: imagen-stage2-trainer, Property 8: Optimizer state round-trip**
  - **Validates: Requirements 7.2**

- [ ] 7. Final Checkpoint — Ensure all tests pass, ask the user if questions arise.
