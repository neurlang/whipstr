"""
Train the Imagen acoustic codec decoder (Stage 2).

Usage:
    python3 imagen_train_example.py

Requires a frozen WhipstrEncoder checkpoint at checkpoints/best_model.pt.
"""

import torch
from imagen import ImagenTrainer, SpectrogramWindowDataset, ImageGenerator

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER_CKPT = "checkpoints/best_model.pt"

print(f"Using device: {DEVICE}")

dataset = SpectrogramWindowDataset(
    source="data/TSV_SPEECH/speech.tsv",
    encoder_checkpoint_path=ENCODER_CKPT,
    limit=9999,
    device=DEVICE,
)

trainer = ImagenTrainer(
    encoder_checkpoint_path=ENCODER_CKPT,
    generator=ImageGenerator(),
    lr=1e-4,
    device=DEVICE,
)

trainer.train(
    dataset=dataset,
    num_epochs=100,
    batch_size=16,
    checkpoint_interval=10,
    output_dir="checkpoints/imagen",
)
