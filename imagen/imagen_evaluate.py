"""
Imagen evaluation: reconstruct a spectrogram from encoder tokens (multi-step DDPM).
Compares original phase spectrogram with the generator's denoised output.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from phase import Phase
from whipstr.whipstr_encoder import WhipstrEncoder
from imagen.image_generator import ImageGenerator

# Config
ENCODER_CKPT = "checkpoints/best_model.pt"
IMAGEN_CKPT = "checkpoints/imagen/checkpoint_epoch_0100.pt"
AUDIO_FILE = "/run/media/m/deepseek/m2/commonvoice_21/cv-corpus-21.0-2025-03-14/af/clips/common_voice_af_38086399.mp3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 11
NUM_STEPS = 1000
OUTPUT_PATH = "imagen_eval.png"


def main():
    print(f"Device: {DEVICE}")

    # Load Encoder
    print("Loading encoder...")
    encoder = WhipstrEncoder(window_size=WINDOW_SIZE)
    ckpt = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.to(DEVICE)
    encoder.eval()

    # Load ImageGenerator
    print("Loading ImageGenerator...")
    imagen_ckpt = torch.load(IMAGEN_CKPT, map_location=DEVICE, weights_only=False)
    gen = ImageGenerator(**imagen_ckpt["config"])
    gen.load_state_dict(imagen_ckpt["generator_state_dict"])
    gen.to(DEVICE)
    gen.eval()

    # Compute phase spectrogram
    print("Computing phase spectrogram...")
    phase = Phase(y_reverse=True)
    audio = phase.to_tensor_flac(AUDIO_FILE)
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio).float()
    else:
        audio = audio.float()

    num_freqs = phase.num_freqs
    total_samples = audio.shape[0]
    W = total_samples // num_freqs
    audio = audio[:W * num_freqs].reshape(W, num_freqs, 2).permute(2, 1, 0)
    spec = audio.float()  # (2, num_freqs, W)

    # Pad/crop to 836 frequencies (matching dataset behavior)
    HEIGHT = 836
    channels, height, width = spec.shape
    if height < HEIGHT:
        pad = torch.zeros(channels, HEIGHT - height, width)
        spec = torch.cat([spec, pad], dim=1)
    elif height > HEIGHT:
        spec = spec[:, :HEIGHT, :]
    spec = spec.contiguous()

    T = spec.shape[2] - WINDOW_SIZE + 1
    print(f"Spectrogram shape: {spec.shape}, windows: {T}")

    # Precompute tokens for all windows
    tokens = []
    for w_idx in range(T):
        window = spec[:, :, w_idx:w_idx + WINDOW_SIZE]
        with torch.no_grad():
            tok = encoder(window.unsqueeze(0).to(DEVICE))
        tokens.append(tok.squeeze(0))
    tokens = torch.cat(tokens, dim=0)  # (T, 64)

    # Shared spectrogram, starts from noise
    X = torch.randn(2, HEIGHT, spec.shape[2], device=DEVICE)
    dt = 1.0 / NUM_STEPS

    # Breath-first multi-step denoising (averaging noise at each step)
    for step in range(NUM_STEPS):
        print(step)
        noise_pred_sum = torch.zeros(2, HEIGHT, spec.shape[2], device=DEVICE)
        noise_pred_cnt = torch.zeros(2, HEIGHT, spec.shape[2], device=DEVICE)

        for w_idx in range(T):
            window = X[:, :, w_idx:w_idx + WINDOW_SIZE].permute(0, 2, 1).reshape(1, 2, WINDOW_SIZE, HEIGHT)
            with torch.no_grad():
                noise_pred = gen(window, tokens[w_idx:w_idx+1])
            np_win = noise_pred[0].reshape(2, WINDOW_SIZE, HEIGHT).permute(0, 2, 1)
            noise_pred_sum[:, :, w_idx:w_idx + WINDOW_SIZE] += np_win
            noise_pred_cnt[:, :, w_idx:w_idx + WINDOW_SIZE] += 1

        noise_avg = noise_pred_sum / noise_pred_cnt.clamp(min=1)
        X -= dt * noise_avg

    reconstructed = X.cpu()

    # Plot
    print("Plotting...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    orig = spec[0].cpu().float()
    recon = reconstructed[0].cpu().float()

    vmin = orig.min().item()
    vmax = orig.max().item()
    norm = mcolors.SymLogNorm(linthresh=0.1, linscale=1, vmin=vmin, vmax=vmax)

    axes[0].imshow(orig, aspect='auto', cmap='viridis', norm=norm)
    axes[0].set_title("Original Phase Spectrogram (channel 0, symlog)")

    rmin = recon.min().item()
    rmax = recon.max().item()
    recon_norm = mcolors.SymLogNorm(linthresh=0.1, linscale=1, vmin=rmin, vmax=rmax)
    axes[1].imshow(recon, aspect='auto', cmap='viridis', norm=recon_norm)
    axes[1].set_title(f"Reconstructed from Encoder Tokens ({NUM_STEPS}-step, symlog)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH} (T={T} windows)")


if __name__ == "__main__":
    main()
