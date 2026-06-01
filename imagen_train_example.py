"""
Minimal working example of training Imagen (Stage 2) as an acoustic codec decoder.

Usage:
    python3 imagen_train_example.py

Requires a WhipstrEncoder checkpoint at checkpoints/best_model.pt.
"""

from imagen import ImagenTrainer, SpectrogramWindowDataset, ImageGenerator
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ENCODER_CKPT = "checkpoints/best_model.pt"

dataset = SpectrogramWindowDataset(
    source="data/TSV_SPEECH/speech.tsv",
    encoder_checkpoint_path=ENCODER_CKPT,
    limit=9999,
    device=device,
)

trainer = ImagenTrainer(
    encoder_checkpoint_path=ENCODER_CKPT,
    generator=ImageGenerator(),
    lr=1e-4,
    device=device,
)

trainer.train(
    dataset=dataset,
    num_epochs=100,
    batch_size=16,
    checkpoint_interval=10,
    output_dir="checkpoints/imagen",
)
