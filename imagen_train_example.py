from imagen import ImagenTrainer, SpectrogramWindowDataset

# 1. Create the dataset from your TSV file
dataset = SpectrogramWindowDataset(source="data/TSV_SPEECH/speech.tsv",limit=99999)

# 2. Create the trainer (needs a Stage 1 encoder checkpoint)
trainer = ImagenTrainer(
    encoder_checkpoint_path="checkpoints/best_model.pt",  # from train_improved.py
    lambda_recon=0.1,
    lr=1e-4,
    device="cpu",  # or "cuda"
)

# 3. Train
trainer.train(
    dataset=dataset,
    num_epochs=100,
    batch_size=16,
    checkpoint_interval=10,
    output_dir="checkpoints/imagen",
)
