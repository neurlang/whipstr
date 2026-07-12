"""
Example usage script for the Whipstr STT (ASR) system with TSV speech data.

This script demonstrates:
1. Dataset creation with TSV speech data
2. Model instantiation
3. Training for a few epochs
4. Inference on a test sample
"""

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
from whipstr.whipstr_encoder import WhipstrEncoder
from whipstr.whipstr_transformer import WhipstrTransformer


def solution_string_to_tensor(solution_strings, char_to_idx, pad_id, eos_id, device):
    """Convert transcription strings to tensor of character indices.

    Builds sequence as: [BOS, chars..., EOS] then pads with PAD to uniform length.
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


def main():
    print("=" * 60)
    print("Whipstr STT (ASR) - Example Usage")
    print("=" * 60)

    # Configuration
    batch_size = 1
    num_epochs = 3600
    learning_rate = 0.00002
    stride = 1
    window_size = 11
    dataset_limit = 500

    # Device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device Setup")
    print(f"   Using device: {device}")

    # Step 1: Create Whipstr STT TSV Speech dataset
    print(f"\n2. Creating Whipstr STT TSV Speech Dataset")

    # Build character vocabulary from full dataset, then restrict to training split
    all_chars = set()

    dataset = WhipstrTSVSpeechDataset(
        tsv_path='data/TSV_SPEECH/speech.tsv',
        limit=dataset_limit,
        all_chars=all_chars,
    )

    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Build vocabulary from training split only
    train_chars = set()
    for idx in train_dataset.indices:
        _, text = dataset[idx]
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

    print(f"   Dataset size: {len(dataset)}")
    print(f"   Training vocabulary size: {len(train_chars)}")
    print(f"   Transformer vocab size (incl. PAD/BOS/UNK + EOS): {transformer_vocab_size}")
    print(f"   Characters: {sorted(train_chars)}")

    # Show a sample
    sample_image, sample_solution = dataset[0]
    print(f"\n   Sample from dataset:")
    print(f"   - Spectrogram shape: {sample_image.shape}")
    print(f"   - Transcription: '{sample_solution}'")
    print(f"   - Spectrogram width: {sample_image.shape[2]} pixels")

    # Step 2: Create dataloaders
    print(f"\n3. Creating DataLoaders")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"   Batch size: {batch_size}")
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")
    print(f"   Number of training batches: {len(train_loader)}")
    print(f"   Number of validation batches: {len(val_loader)}")

    # Step 3: Initialize models
    print(f"\n4. Initializing Models")

    encoder = WhipstrEncoder(stride=stride, window_size=window_size).to(device)
    print(f"   CNN Encoder:")
    print(f"   - Stride: {stride}")
    print(f"   - Window size: {window_size}x{window_size}")

    transformer = WhipstrTransformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        vocab_size=transformer_vocab_size
    ).to(device)
    print(f"   Transformer:")
    print(f"   - Model dimension: 256")
    print(f"   - Attention heads: 8")
    print(f"   - Encoder layers: 4")
    print(f"   - Decoder layers: 4")
    print(f"   - Vocabulary size: {transformer_vocab_size}")

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    transformer_params = sum(p.numel() for p in transformer.parameters())
    print(f"\n   Total parameters:")
    print(f"   - Encoder: {encoder_params:,}")
    print(f"   - Transformer: {transformer_params:,}")
    print(f"   - Total: {encoder_params + transformer_params:,}")

    # Step 4: Setup optimizer and loss
    print(f"\n5. Setting Up Training")
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(transformer.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    print(f"   Optimizer: Adam")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Loss function: CrossEntropyLoss (ignore_index={pad_id})")

    # Step 5: Training loop
    print(f"\n6. Training for {num_epochs} Epochs")
    print("   " + "-" * 56)

    for epoch in range(num_epochs):
        # Training
        encoder.train()
        transformer.train()

        train_loss = 0.0
        num_batches = 0

        for batch_idx, (images, solution_strings) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            targets = solution_string_to_tensor(solution_strings, char_to_idx, pad_id, eos_id, device)

            optimizer.zero_grad()

            # Forward pass
            encoder_tokens = encoder(images)
            decoder_input = targets[:, :-1]
            labels = targets[:, 1:]
            causal_mask = transformer._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            target_padding_mask = decoder_input.eq(pad_id)
            logits = transformer(encoder_tokens, decoder_input, target_mask=causal_mask, target_padding_mask=target_padding_mask)

            # Compute loss
            loss = criterion(logits.transpose(1, 2), labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Validation
        encoder.eval()
        transformer.eval()

        val_loss = 0.0
        total_correct = 0
        total_chars = 0
        num_val_batches = 0

        with torch.no_grad():
            for images, solution_strings in val_loader:
                images = images.to(device)
                targets = solution_string_to_tensor(solution_strings, char_to_idx, pad_id, eos_id, device)

                encoder_tokens = encoder(images)
                decoder_input = targets[:, :-1]
                labels = targets[:, 1:]
                causal_mask = transformer._generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                target_padding_mask = decoder_input.eq(pad_id)
                logits = transformer(encoder_tokens, decoder_input, target_mask=causal_mask, target_padding_mask=target_padding_mask)

                loss = criterion(logits.transpose(1, 2), labels)
                val_loss += loss.item()
                num_val_batches += 1

                predictions = torch.argmax(logits, dim=-1)
                valid_mask = labels.ne(pad_id)
                correct = ((predictions == labels) & valid_mask).sum().item()
                total_correct += correct
                total_chars += valid_mask.sum().item()

        avg_val_loss = val_loss / max(num_val_batches, 1)
        val_accuracy = total_correct / max(total_chars, 1)

        print(f"   Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Acc={val_accuracy:.4f}")

    print("   " + "-" * 56)
    print("   Training complete!")

    # Step 6: Inference on a test sample
    print(f"\n7. Inference on Test Sample")

    encoder.eval()
    transformer.eval()

    test_idx = 0 if len(val_dataset) > 0 else 0
    test_image, test_solution = val_dataset[test_idx] if len(val_dataset) > 0 else dataset[0]
    print(f"   Ground truth: '{test_solution}'")

    test_image_batch = test_image.unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_tokens = encoder(test_image_batch)
        print(f"   Encoder output shape: {encoder_tokens.shape}")

        max_length = max(len(s) for _, s in dataset)
        predictions = transformer.generate(
            encoder_tokens,
            max_length=max_length,
            start_token=pad_id,
            eos_token=eos_id,
        )

        predicted_indices = predictions[0].cpu().tolist()
        # Stop at first EOS token
        if eos_id in predicted_indices:
            predicted_indices = predicted_indices[:predicted_indices.index(eos_id)]
        predicted_string = ''.join(idx_to_char.get(idx, '?') for idx in predicted_indices if idx > 0 and idx < eos_id)

        print(f"   Predicted:    '{predicted_string}'")

        if predicted_string == test_solution:
            print(f"   ✓ Prediction matches ground truth!")
        else:
            correct_count = sum(1 for p, t in zip(predicted_string, test_solution) if p == t)
            print(f"   Correct characters: {correct_count}/{len(test_solution)}")

    print(f"\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
