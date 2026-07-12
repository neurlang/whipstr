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

    Pads spectrograms to the maximum width in the batch and returns
    original widths for constructing the encoder padding mask.
    """
    images, solutions = zip(*batch)
    widths = torch.tensor([img.shape[2] for img in images])
    max_width = widths.max().item()
    padded_images = []
    for img in images:
        if img.shape[2] < max_width:
            padding = torch.zeros(2, img.shape[1], max_width - img.shape[2])
            padded_img = torch.cat([img, padding], dim=2)
        else:
            padded_img = img
        padded_images.append(padded_img)
    images_tensor = torch.stack(padded_images, dim=0)
    return images_tensor, solutions, widths


def solution_string_to_tensor(solution_strings, char_to_idx, pad_id, eos_id, device):
    """Convert transcription strings to tensor of character indices.

    Builds sequence as: [BOS, chars..., EOS] then pads with PAD to uniform length.

    BOS and PAD share ID 0, so BOS positions will also be masked in
    target_padding_mask — the first position relies on cross-attention to
    encoder memory and residual connections.

    Args:
        solution_strings: List of transcription strings
        char_to_idx: Dict mapping characters to indices
        pad_id: ID for padding (also used as BOS and UNK)
        eos_id: ID for end-of-sequence token
        device: torch device (CPU or CUDA)

    Returns:
        torch.LongTensor of shape [batch, max_seq_len] with structure
        [BOS, c1, c2, ..., cn, EOS, PAD, ...]
    """
    bos_id = pad_id
    unk_id = pad_id
    char_lists = [[char_to_idx.get(c, unk_id) for c in s] for s in solution_strings]
    max_text_len = max(len(c) for c in char_lists)
    total_len = max_text_len + 2  # BOS + chars + EOS
    padded = []
    for chars in char_lists:
        seq = [bos_id] + chars + [eos_id]
        seq += [pad_id] * (total_len - len(seq))
        padded.append(seq)
    return torch.tensor(padded, dtype=torch.long, device=device)


def train_epoch(encoder, transformer, dataloader, optimizer, criterion, device, char_to_idx, pad_id, eos_id, window_size, clip_grad=1.0):
    """Train for one epoch.

    Args:
        encoder: WhipstrEncoder model
        transformer: WhipstrTransformer model
        dataloader: DataLoader for training data
        optimizer: Optimizer for both models
        criterion: Loss function (CrossEntropyLoss with ignore_index=pad_id)
        device: torch device (CPU or CUDA)
        char_to_idx: Dict mapping characters to indices
        pad_id: ID for padding/BOS/UNK token
        eos_id: ID for end-of-sequence token
        window_size: Window size used by the encoder (for mask computation)
        clip_grad: Gradient clipping value (default: 1.0)

    Returns:
        float: Average loss for the epoch
    """
    encoder.train()
    transformer.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, solution_strings, widths) in enumerate(dataloader):
        images = images.to(device)
        widths = widths.to(device)
        targets = solution_string_to_tensor(solution_strings, char_to_idx, pad_id, eos_id, device)

        optimizer.zero_grad()

        # Forward pass
        encoder_tokens = encoder(images)
        # Build encoder padding mask: True at positions beyond the sample's real windows
        T = encoder_tokens.shape[1]
        real_windows = widths - window_size + 1
        encoder_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= real_windows.unsqueeze(1)
        # decoder_input: BOS + all chars (exclude last position)
        # labels: all chars + EOS (exclude first position)
        decoder_input = targets[:, :-1]
        labels = targets[:, 1:]
        causal_mask = transformer._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
        target_padding_mask = decoder_input.eq(pad_id)
        logits = transformer(
            encoder_tokens, decoder_input,
            target_mask=causal_mask,
            encoder_padding_mask=encoder_padding_mask,
            target_padding_mask=target_padding_mask,
        )

        # Compute loss (ignore PAD positions)
        loss = criterion(logits.transpose(1, 2), labels)

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


def validate(encoder, transformer, dataloader, criterion, device, char_to_idx, pad_id, eos_id, window_size):
    """Evaluate on validation set.

    Args:
        encoder: WhipstrEncoder model
        transformer: WhipstrTransformer model
        dataloader: DataLoader for validation data
        criterion: Loss function (CrossEntropyLoss with ignore_index=pad_id)
        device: torch device (CPU or CUDA)
        char_to_idx: Dict mapping characters to indices
        pad_id: ID for padding/BOS/UNK token
        eos_id: ID for end-of-sequence token
        window_size: Window size used by the encoder (for mask computation)

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
        for images, solution_strings, widths in dataloader:
            images = images.to(device)
            widths = widths.to(device)
            targets = solution_string_to_tensor(solution_strings, char_to_idx, pad_id, eos_id, device)

            # Forward pass
            encoder_tokens = encoder(images)
            # Build encoder padding mask: True at positions beyond the sample's real windows
            T = encoder_tokens.shape[1]
            real_windows = widths - window_size + 1
            encoder_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= real_windows.unsqueeze(1)
            decoder_input = targets[:, :-1]
            labels = targets[:, 1:]
            causal_mask = transformer._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            target_padding_mask = decoder_input.eq(pad_id)
            logits = transformer(
                encoder_tokens, decoder_input,
                target_mask=causal_mask,
                encoder_padding_mask=encoder_padding_mask,
                target_padding_mask=target_padding_mask,
            )

            # Compute loss (ignore PAD positions)
            loss = criterion(logits.transpose(1, 2), labels)

            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy over non-padded positions only
            predictions = torch.argmax(logits, dim=-1)
            valid_mask = labels.ne(pad_id)
            correct = ((predictions == labels) & valid_mask).sum().item()
            total_correct += correct
            total_chars += valid_mask.sum().item()

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

    # Build character vocabulary from the full dataset first to determine split
    all_chars = set()

    full_dataset = WhipstrTSVSpeechDataset(
        tsv_path='data/TSV_SPEECH/speech.tsv',
        limit=dataset_limit,
        all_chars=all_chars,
    )

    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Build vocabulary from training split only
    train_chars = set()
    for idx in train_dataset.indices:
        _, text = full_dataset[idx]
        train_chars.update(text)

    # Token ID assignment:
    #   0 = PAD / BOS / UNK  (shared)
    #   1..N = real characters
    #   N+1 = EOS
    char_to_idx = {char: i + 1 for i, char in enumerate(sorted(train_chars))}
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    pad_id = 0
    eos_id = len(char_to_idx) + 1
    transformer_vocab_size = eos_id + 1  # 0 + N chars + 1 for EOS

    print(f"Dataset size: {len(full_dataset)}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Training vocabulary size: {len(train_chars)}")
    print(f"Transformer vocab size (incl. PAD/BOS/UNK + EOS): {transformer_vocab_size}")

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
        vocab_size=transformer_vocab_size
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

    # Loss function (ignore PAD token at index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, transformer, train_loader, optimizer, criterion, device, char_to_idx, pad_id, eos_id, window_size)
        val_loss, val_accuracy = validate(encoder, transformer, val_loader, criterion, device, char_to_idx, pad_id, eos_id, window_size)

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
                'pad_id': pad_id,
                'eos_id': eos_id,
                'vocab_size': transformer_vocab_size,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete!")


if __name__ == '__main__':
    main()
