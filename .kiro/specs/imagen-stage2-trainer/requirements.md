# Requirements Document

## Introduction

Stage 2 of the Whipstr pipeline introduces an image generator trainer. The frozen Stage 1 CNN encoder (`WhipstrEncoder`) is repurposed as the **decoder** half of an autoencoder. A new asymmetric 2D U-Net model is trained as the **encoder** (generator) half: it takes a spectrogram window of shape `(2, 11, 836)` and reconstructs the same window. The generator uses a diffusion-style noise-prediction objective conditioned on a 10-float conditioning vector. All training code lives in the `imagen/` folder.

## Glossary

- **Autoencoder**: A neural network that learns to encode an input into a compressed representation and then decode it back to the original input.
- **Stage 1 CNN**: The `WhipstrEncoder` model from `whipstr/whipstr_encoder.py`, trained in Stage 1 for ASR. Used frozen as the decoder in Stage 2.
- **Image Generator (UNet)**: The new asymmetric 2D U-Net model trained in Stage 2 to reconstruct spectrogram windows.
- **Spectrogram Window**: A tensor of shape `(2, 11, 836)` — 2 channels (phase), 11 time frames, 836 frequency bins.
- **Bottleneck**: The compressed latent representation inside the U-Net, shape `(channels, 1, freq_reduced)` after collapsing the time axis.
- **Conditioning Vector**: A 64-float vector fed into the bottleneck via a small MLP to guide generation.
- **Diffusion Step**: A denoising step where the model predicts the noise added to a noisy input `x_t` to recover `x_{t-1}`.
- **Noise Prediction**: The U-Net output — a tensor of shape `(2, 11, 836)` representing predicted noise.
- **MLP**: Multi-Layer Perceptron — a small fully-connected network used to project the conditioning vector.
- **ConvTranspose2d**: Transposed convolution used for upsampling in the decoder path of the U-Net.
- **Skip Connection**: A direct connection from an encoder layer to the corresponding decoder layer in the U-Net.
- **Checkpoint**: A saved file containing model weights and optimizer state at a given training step.
- **MSE Loss**: Mean Squared Error loss, used to measure reconstruction quality.
- **Frozen Model**: A model whose parameters are not updated during training (gradients disabled).

## Requirements

### Requirement 1

**User Story:** As a researcher, I want an asymmetric 2D U-Net image generator model, so that I can learn to reconstruct spectrogram windows from a compressed bottleneck.

#### Acceptance Criteria

1. THE ImageGenerator SHALL accept an input tensor of shape `(batch, 2, 11, 836)` and produce an output tensor of the same shape `(batch, 2, 11, 836)`.
2. WHEN the time axis is downsampled, THE ImageGenerator SHALL collapse 11 time frames to 1 using a single `Conv2d` with `kernel_size=(11,1)` and `stride=(11,1)`.
3. WHEN the frequency axis is downsampled, THE ImageGenerator SHALL apply at least 3 sequential stride-2 convolutions along the frequency axis, reducing 836 to at most 105 frequency bins at the bottleneck.
4. WHEN the frequency axis is upsampled, THE ImageGenerator SHALL use `ConvTranspose2d` layers to restore the frequency dimension back to 836.
5. WHEN the time axis is upsampled, THE ImageGenerator SHALL restore 11 time frames from 1 using `ConvTranspose2d` with `kernel_size=(11,1)` and `stride=(11,1)`.
6. THE ImageGenerator SHALL include skip connections between matching encoder and decoder frequency levels.

### Requirement 2

**User Story:** As a researcher, I want the conditioning vector to influence generation, so that the model can produce context-aware reconstructions.

#### Acceptance Criteria

1. THE ImageGenerator SHALL accept a conditioning vector of exactly 64 floats per sample as a second input.
2. WHEN the conditioning vector is provided, THE ImageGenerator SHALL project it through an MLP to a 128-dimensional vector and add it to the bottleneck feature map.
3. THE MLP SHALL consist of at least 2 linear layers with a non-linear activation between them.

### Requirement 3

**User Story:** As a researcher, I want the Stage 1 CNN to be used as the frozen decoder, so that the autoencoder leverages already-learned acoustic features.

#### Acceptance Criteria

1. WHEN Stage 2 training begins, THE Trainer SHALL load the Stage 1 `WhipstrEncoder` weights from a checkpoint file path provided as a configuration parameter.
2. WHEN the Stage 1 model is loaded, THE Trainer SHALL freeze all parameters of the `WhipstrEncoder` so that its weights are not updated during Stage 2 training.
3. THE Trainer SHALL pass each reconstructed spectrogram window through the frozen `WhipstrEncoder` to obtain a token sequence, and use the token sequence as part of the training signal.

### Requirement 4

**User Story:** As a researcher, I want a diffusion-style noise-prediction training objective, so that the image generator learns to denoise spectrogram windows.

#### Acceptance Criteria

1. WHEN a training batch is processed, THE Trainer SHALL add Gaussian noise scaled by a randomly sampled noise level `t ∈ [0, 1]` to each spectrogram window to produce a noisy input `x_t`.
2. WHEN the noisy input is passed through the ImageGenerator, THE Trainer SHALL compute the MSE loss between the predicted noise and the actual noise that was added.
3. THE Trainer SHALL also compute an auxiliary reconstruction loss by passing the denoised prediction through the frozen `WhipstrEncoder` and comparing its token output to the token output of the clean input.
4. THE Trainer SHALL combine the noise-prediction MSE loss and the auxiliary reconstruction loss using a configurable weighting parameter `lambda_recon`.

### Requirement 5

**User Story:** As a researcher, I want a training loop with checkpointing and logging, so that I can monitor and resume training.

#### Acceptance Criteria

1. THE Trainer SHALL train for a configurable number of epochs over a dataset of spectrogram windows.
2. WHEN an epoch completes, THE Trainer SHALL log the average noise-prediction loss and auxiliary reconstruction loss to stdout.
3. WHEN a checkpoint interval is reached, THE Trainer SHALL save the `ImageGenerator` weights, optimizer state, and current epoch to a `.pt` file in a configurable output directory.
4. WHEN training is resumed, THE Trainer SHALL load a checkpoint and continue training from the saved epoch.

### Requirement 6

**User Story:** As a researcher, I want a dataset class for spectrogram windows, so that the trainer can load and iterate over training data efficiently.

#### Acceptance Criteria

1. THE SpectrogramWindowDataset SHALL load spectrogram tensors of shape `(2, 836, W)` from a source compatible with the existing `WhipstrTSVSpeechDataset` format.
2. WHEN a sample is requested, THE SpectrogramWindowDataset SHALL extract non-overlapping windows of shape `(2, 11, 836)` from the full spectrogram.
3. WHEN a spectrogram width is not evenly divisible by 11, THE SpectrogramWindowDataset SHALL discard the remainder frames.
4. THE SpectrogramWindowDataset SHALL return a conditioning vector of 64 floats alongside each window, derived from the window's position and audio statistics.

### Requirement 7

**User Story:** As a researcher, I want the image generator and trainer to be serializable and deserializable, so that I can save and reload models reliably.

#### Acceptance Criteria

1. WHEN the `ImageGenerator` model is serialized using `torch.save` and deserialized using `torch.load`, THE system SHALL produce a model with identical weights and architecture.
2. WHEN a checkpoint is saved and reloaded, THE Trainer SHALL restore the optimizer state and epoch counter to their exact saved values.
3. THE `ImageGenerator` SHALL expose a `get_config` method that returns a dictionary of all constructor hyperparameters, enabling reconstruction of the model from config alone.
