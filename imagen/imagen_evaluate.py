"""
Imagen evaluation: reconstruct a full spectrogram from encoder tokens via DDPM.

Algorithm
---------
1. Load audio → phase spectrogram (2, 836, W).
2. Slide the frozen encoder over all overlapping windows → tokens (T, 64).
3. Initialise the full canvas X ~ N(0, data_std²·I), shape (2, 836, W).
4. For each reverse diffusion step s = T-1 … 0:
     a. For every overlapping window position w:
          - Extract window X[:, :, w:w+11]  shape (2, 836, 11)
          - Predict noise:  ε_w = generator(window, token_w, t=s)
     b. Average overlapping ε predictions into a full-canvas noise estimate ε̄.
     c. Apply one DDPM reverse step:
          x_{s-1} = (x_s - sqrt(1-ᾱ_s)/sqrt(ᾱ_s) * ε̄) / sqrt(α_s/ᾱ_{s-1})
          + σ_s * z   (z ~ N(0,I), omitted at s=0)
5. Plot original vs reconstructed.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Config — edit these before running
ENCODER_CKPT = "checkpoints/best_model.pt"
IMAGEN_CKPT  = "checkpoints/imagen/checkpoint_epoch_0100.pt"
AUDIO_FILE   = "/home/m/Downloads/LJ001-0001.wav"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE  = 11
# T_START: noise level to begin reverse diffusion from.
# 999 = pure noise (ignore template), 0 = pure template (no diffusion).
# Values around 800-900 give a good template-guided start.
T_START      = 999
OUTPUT_PATH  = "imagen_eval.png"


# ---------------------------------------------------------------------------
# Schedule helpers (must match training)
# ---------------------------------------------------------------------------

def _make_linear_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas     = torch.linspace(beta_start, beta_end, num_steps)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # ── Load encoder ──────────────────────────────────────────────────────
    print("Loading encoder...")
    from whipstr.whipstr_encoder import WhipstrEncoder
    ckpt = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("encoder_state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state_dict = ckpt
    encoder = WhipstrEncoder(window_size=WINDOW_SIZE)
    encoder.load_state_dict(state_dict)
    encoder.to(DEVICE)
    encoder.eval()

    # ── Load generator ────────────────────────────────────────────────────
    print("Loading ImageGenerator...")
    from imagen.image_generator import ImageGenerator
    gen_ckpt = torch.load(IMAGEN_CKPT, map_location=DEVICE, weights_only=False)
    gen = ImageGenerator(**gen_ckpt["config"])
    gen.load_state_dict(gen_ckpt["generator_state_dict"])
    gen.to(DEVICE)
    gen.eval()

    T = gen.num_steps

    # ── Load audio → spectrogram ──────────────────────────────────────────
    print("Computing phase spectrogram...")
    from phase import Phase
    phase = Phase(y_reverse=True)
    audio = phase.to_tensor_flac(AUDIO_FILE)
    if not isinstance(audio, torch.Tensor):
        import torch as _t
        audio = _t.from_numpy(audio).float()
    else:
        audio = audio.float()

    num_freqs = phase.num_freqs
    W_frames  = audio.shape[0] // num_freqs
    audio     = audio[:W_frames * num_freqs].reshape(W_frames, num_freqs, 2).permute(2, 1, 0)
    # (2, num_freqs, W_frames)

    HEIGHT = 836
    ch, h, w = audio.shape
    if h < HEIGHT:
        audio = torch.cat([audio, torch.zeros(ch, HEIGHT - h, w)], dim=1)
    elif h > HEIGHT:
        audio = audio[:, :HEIGHT, :]
    spec = audio.float().contiguous()  # (2, 836, W)
    W    = spec.shape[2]
    N    = W - WINDOW_SIZE + 1         # number of overlapping windows
    print(f"Spectrogram: {spec.shape}, windows: {N}")

    # ── Precompute encoder tokens ─────────────────────────────────────────
    print("Computing encoder tokens...")
    with torch.no_grad():
        tokens = encoder(spec.unsqueeze(0).to(DEVICE))  # (1, N, 64)
    tokens = tokens.squeeze(0)  # (N, 64)

    # ── Build noise schedule ──────────────────────────────────────────────
    betas, alphas, alpha_bar = _make_linear_schedule(T)
    betas     = betas.to(DEVICE)
    alphas    = alphas.to(DEVICE)
    alpha_bar = alpha_bar.to(DEVICE)

    # alpha_bar_{t-1}: prepend 1.0 for t=0
    alpha_bar_prev = torch.cat([torch.ones(1, device=DEVICE), alpha_bar[:-1]], dim=0)

    # Posterior variance σ²_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
    posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp(min=1e-8)

    sqrt_ab   = alpha_bar.sqrt()
    sqrt_1mab = (1.0 - alpha_bar).sqrt()

    # ── Initialise full canvas via make_start ─────────────────────────────
    t_start = min(T_START, T - 1)
    print(f"Starting from t_start={t_start} ({'pure noise' if t_start == T-1 else 'template-guided'})")

    # Each overlapping window position gets its own make_start sample,
    # then we average the overlapping contributions into a single canvas.
    X       = torch.zeros(2, HEIGHT, W, device=DEVICE)
    X_cnt   = torch.zeros(2, HEIGHT, W, device=DEVICE)
    with torch.no_grad():
        for w_idx in range(N):
            Xw = gen.make_start(1, t_start, sqrt_ab, sqrt_1mab, torch.device(DEVICE))
            X    [:, :, w_idx : w_idx + WINDOW_SIZE] += Xw.squeeze(0)
            X_cnt[:, :, w_idx : w_idx + WINDOW_SIZE] += 1.0
    X = X / X_cnt.clamp(min=1.0)

    # ── Reverse diffusion loop ────────────────────────────────────────────
    print(f"Running {T}-step reverse diffusion...")
    with torch.no_grad():
        for s in reversed(range(T)):
            if s % 100 == 0:
                print(f"  step {s}")

            t_tensor = torch.full((1,), s, dtype=torch.long, device=DEVICE)

            # Accumulate noise predictions over all overlapping windows
            eps_sum = torch.zeros(2, HEIGHT, W, device=DEVICE)
            eps_cnt = torch.zeros(2, HEIGHT, W, device=DEVICE)

            for w_idx in range(N):
                window = X[:, :, w_idx : w_idx + WINDOW_SIZE].unsqueeze(0)  # (1,2,836,11)
                tok    = tokens[w_idx : w_idx + 1]                           # (1,64)

                eps_w = gen(window, tok, t_tensor)                           # (1,2,836,11)
                eps_sum[:, :, w_idx : w_idx + WINDOW_SIZE] += eps_w.squeeze(0)
                eps_cnt[:, :, w_idx : w_idx + WINDOW_SIZE] += 1.0

            eps_avg = eps_sum / eps_cnt.clamp(min=1.0)  # (2, 836, W)

            # DDPM reverse step
            # x_{s-1} = 1/sqrt(α_s) * (x_s - β_s/sqrt(1-ᾱ_s) * ε̄) + σ_s * z
            coef     = betas[s] / (1.0 - alpha_bar[s]).sqrt().clamp(min=1e-8)
            x_prev   = (X - coef * eps_avg) / alphas[s].sqrt()

            if s > 0:
                noise  = torch.randn_like(X)
                x_prev = x_prev + posterior_var[s].sqrt() * noise

            X = x_prev

    reconstructed = X.cpu()

    # ── Plot ──────────────────────────────────────────────────────────────
    print("Plotting...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    orig  = spec[0].cpu().float()
    recon = reconstructed[0].cpu().float()

    # Shared normalisation so both plots are directly comparable
    vmin = orig.min().item()
    vmax = orig.max().item()
    norm = mcolors.SymLogNorm(linthresh=0.1, linscale=1, vmin=vmin, vmax=vmax)

    axes[0].imshow(orig,  aspect="auto", cmap="viridis", norm=norm)
    axes[0].set_title("Original Phase Spectrogram (channel 0)")
    axes[0].set_ylabel("Frequency bin")

    axes[1].imshow(recon, aspect="auto", cmap="viridis", norm=norm)
    axes[1].set_title(f"Reconstructed from Encoder Tokens ({T}-step DDPM, channel 0)")
    axes[1].set_ylabel("Frequency bin")
    axes[1].set_xlabel("Time frame")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
