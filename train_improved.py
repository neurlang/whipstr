"""Improved Whipstr STT (ASR) training script with better hyperparameters and monitoring."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
from whipstr.whipstr_encoder import WhipstrEncoder
from whipstr.whipstr_transformer import WhipstrTransformer
import os


def collate_fn(batch):
    """Custom collate function to handle variable-width spectrograms."""
    images, solutions = zip(*batch)
    max_width = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        if img.shape[2] < max_width:
            padding = torch.zeros(2, 840, max_width - img.shape[2])
            padded_img = torch.cat([img, padding], dim=2)
        else:
            padded_img = img
        padded_images.append(padded_img)
    images_tensor = torch.stack(padded_images, dim=0)
    return images_tensor, solutions


def solution_string_to_tensor(solution_strings, char_to_idx, device):
    """Convert transcription strings to tensor of character indices."""
    char_lists = [[char_to_idx.get(c, 0) for c in s] for s in solution_strings]
    max_len = max(len(c) for c in char_lists)
    padded = []
    for chars in char_lists:
        padded_seq = chars + [0] * (max_len - len(chars))
        padded.append(padded_seq)
    return torch.tensor(padded, dtype=torch.long, device=device)


def train_epoch(encoder, transformer, dataloader, optimizer, criterion, device, char_to_idx, vocab_size, start_token_idx, clip_grad=1.0):
    """Train for one epoch."""
    encoder.train()
    transformer.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, solution_strings) in enumerate(dataloader):
        images = images.to(device)
        targets = solution_string_to_tensor(solution_strings, char_to_idx, device)
        batch_size, seq_len = targets.shape

        optimizer.zero_grad()

        # Forward pass
        encoder_tokens = encoder(images)
        start_tokens = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=device)
        decoder_input = torch.cat([start_tokens, targets[:, :-1]], dim=1)
        logits = transformer(encoder_tokens, decoder_input)

        # Compute loss
        logits_flat = logits.reshape(-1, vocab_size + 1)
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)

        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss detected at batch {batch_idx}")

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(encoder, transformer, dataloader, criterion, device, char_to_idx, vocab_size, start_token_idx):
    """Evaluate on validation set."""
    encoder.eval()
    transformer.eval()

    total_loss = 0.0
    total_correct = 0
    total_chars = 0
    num_batches = 0

    with torch.no_grad():
        for images, solution_strings in dataloader:
            images = images.to(device)
            targets = solution_string_to_tensor(solution_strings, char_to_idx, device)
            batch_size, seq_len = targets.shape

            # Forward pass
            encoder_tokens = encoder(images)
            start_tokens = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=device)
            decoder_input = torch.cat([start_tokens, targets[:, :-1]], dim=1)
            logits = transformer(encoder_tokens, decoder_input)

            # Compute loss
            logits_flat = logits.reshape(-1, vocab_size + 1)
            targets_flat = targets.reshape(-1)
            loss = criterion(logits_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == targets).sum().item()
            total_correct += correct
            total_chars += batch_size * seq_len

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = total_correct / total_chars if total_chars > 0 else 0.0

    return avg_loss, accuracy


def main():
    print("=" * 70)
    print("IMPROVED WHIPSTR STT (ASR) TRAINING")
    print("=" * 70)

    # Hyperparameters
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0003
    stride = 1
    window_size = 11
    dataset_limit = 10630

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Initial learning rate: {learning_rate}")
    print(f"  Dataset limit: {dataset_limit}")

    # Build character vocabulary
    all_chars = set()

    dataset = WhipstrTSVSpeechDataset(
        tsv_path='data/TSV_SPEECH/speech.tsv',
        limit=dataset_limit,
        all_chars=all_chars,
    )

    # Create char to index mapping (0 is reserved for padding)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(all_chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    idx_to_char[0] = '<PAD>'
    vocab_size = len(char_to_idx) + 1  # +1 for padding
    start_token_idx = vocab_size  # Use vocab_size as start token

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Characters: {sorted(all_chars)}")

    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Initialize models
    print("\nInitializing models...")
    encoder = WhipstrEncoder(stride=stride, window_size=window_size).to(device)
    transformer = WhipstrTransformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        vocab_size=vocab_size + 1  # +1 for start token
    ).to(device)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    transformer_params = sum(p.numel() for p in transformer.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
    print(f"Total parameters: {encoder_params + transformer_params:,}")

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(transformer.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 30
    checkpoint_dir = './checkpoints'

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(encoder, transformer, train_loader, optimizer, criterion, device, char_to_idx, vocab_size, start_token_idx)

        # Validate
        val_loss, val_accuracy = validate(encoder, transformer, val_loader, criterion, device, char_to_idx, vocab_size, start_token_idx)

        # Step the scheduler
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0

            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'transformer_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'vocab_size': vocab_size,
            }, checkpoint_path)
            print(f"  → New best model saved! Accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break

        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'transformer_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'vocab_size': vocab_size,
            }, checkpoint_path)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
