"""Training pipeline for Whipstr STT (ASR) system.

Integrates the CNN encoder and transformer model for end-to-end training
with backpropagation through both components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
from whipstr.whipstr_encoder import WhipstrEncoder
from whipstr.whipstr_transformer import WhipstrTransformer
import os


def collate_fn(batch):
    """Custom collate function to handle variable-width spectrograms.

    Pads spectrograms to the maximum width in the batch.
    """
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
    """Convert transcription strings to tensor of character indices.

    Args:
        solution_strings: List of transcription strings
        char_to_idx: Dict mapping characters to indices
        device: torch device (CPU or CUDA)

    Returns:
        torch.LongTensor of shape [batch, max_length] with character indices
    """
    char_lists = [[char_to_idx.get(c, 0) for c in s] for s in solution_strings]
    max_len = max(len(c) for c in char_lists)
    padded = []
    for chars in char_lists:
        padded_seq = chars + [0] * (max_len - len(chars))
        padded.append(padded_seq)
    return torch.tensor(padded, dtype=torch.long, device=device)


def train_epoch(encoder, transformer, dataloader, optimizer, criterion, device, char_to_idx, vocab_size, start_token_idx, clip_grad=1.0):
    """Train for one epoch.

    Args:
        encoder: WhipstrEncoder model
        transformer: WhipstrTransformer model
        dataloader: DataLoader for training data
        optimizer: Optimizer for both models
        criterion: Loss function (CrossEntropyLoss)
        device: torch device (CPU or CUDA)
        char_to_idx: Dict mapping characters to indices
        vocab_size: Size of the vocabulary (including padding)
        start_token_idx: Index of the start token
        clip_grad: Gradient clipping value (default: 1.0)

    Returns:
        float: Average loss for the epoch
    """
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
    """Evaluate on validation set.

    Args:
        encoder: WhipstrEncoder model
        transformer: WhipstrTransformer model
        dataloader: DataLoader for validation data
        criterion: Loss function (CrossEntropyLoss)
        device: torch device (CPU or CUDA)
        char_to_idx: Dict mapping characters to indices
        vocab_size: Size of the vocabulary (including padding)
        start_token_idx: Index of the start token

    Returns:
        tuple: (average_loss, accuracy)
    """
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
    """Main training function."""
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    stride = 1
    window_size = 11
    dataset_limit = 10630

    # Device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {vocab_size}")

    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize models
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

    # Optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(transformer.parameters()),
        lr=learning_rate
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Loss function (ignore padding token at index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, transformer, train_loader, optimizer, criterion, device, char_to_idx, vocab_size, start_token_idx)
        val_loss, val_accuracy = validate(encoder, transformer, val_loader, criterion, device, char_to_idx, vocab_size, start_token_idx)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"LR: {current_lr:.6f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = './checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'transformer_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'vocab_size': vocab_size,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete!")


if __name__ == '__main__':
    main()
