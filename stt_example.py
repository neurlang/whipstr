"""
Example usage script for the Whipstr STT (ASR) system with TSV speech data.

This script demonstrates:
1. Dataset creation with TSV speech data
2. Model instantiation
3. Training for a few epochs
4. Inference on a test sample
"""

import argparse
import json
import os
import shutil

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
from whipstr.whipstr_encoder import WhipstrEncoder
from whipstr.whipstr_transformer import WhipstrTransformer


def save_vocab(vocab_list, path):
    """Save vocabulary to a model.json file."""
    with open(path, 'w') as f:
        json.dump({"Vocab": vocab_list}, f, indent=2)


def save_checkpoint(encoder, transformer, optimizer, epoch, val_accuracy, save_dir, vocab_list):
    """Save model checkpoint and vocab to the given directory."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'transformer_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
    }, os.path.join(save_dir, 'checkpoint.pt'))
    save_vocab(vocab_list, os.path.join(save_dir, 'model.json'))


def solution_string_to_tensor(solution_strings, char_to_idx, device):
    """Convert transcription strings to tensor of character indices."""
    char_lists = [[char_to_idx.get(c, 0) for c in s] for s in solution_strings]
    max_len = max(len(c) for c in char_lists)
    padded = []
    for chars in char_lists:
        padded_seq = chars + [0] * (max_len - len(chars))
        padded.append(padded_seq)
    return torch.tensor(padded, dtype=torch.long, device=device)


def collate_fn(batch):
    """Custom collate function to handle variable-width spectrograms.
    
    Pads spectrograms to the maximum width in the batch and returns
    original widths for constructing the encoder padding mask.
    """
    images, solutions = zip(*batch)
    widths = torch.tensor([img.shape[2] for img in images])
    max_width = widths.max().item()
    
    # Pad images to max width
    padded_images = []
    for img in images:
        if img.shape[2] < max_width:
            padding = torch.zeros(2, img.shape[1], max_width - img.shape[2])
            padded_img = torch.cat([img, padding], dim=2)
        else:
            padded_img = img
        padded_images.append(padded_img)
    
    # Stack images
    images_tensor = torch.stack(padded_images, dim=0)
    
    return images_tensor, solutions, widths


def main():
    parser = argparse.ArgumentParser(description="Whipstr STT (ASR) - TSV Speech Example")
    parser.add_argument('--continue-pt', type=str, default=None,
                        help='Path to a .pt checkpoint file to resume training from')
    parser.add_argument('--finetune-pt', type=str, default=None,
                        help='Path to a .pt checkpoint for finetuning (ignores vocabulary size mismatches)')
    args = parser.parse_args()

    print("=" * 60)
    print("Whipstr STT (ASR) - TSV Speech Example")
    print("=" * 60)
    
    # Configuration
    batch_size = 1  # Smaller batch size for faster demo
    num_epochs = 3600  # Training epochs
    learning_rate = 0.00002
    stride = 1  # Stride for CNN encoder
    window_size = 11  # Window size for encoder
    
    # Device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device Setup")
    print(f"   Using device: {device}")
    
    # Step 1: Create Whipstr STT TSV Speech dataset
    print(f"\n2. Creating Whipstr STT TSV Speech Dataset")
    print(f"   Configuration:")
    print(f"   - limit: {1000}")
    
    # Build character vocabulary
    all_chars = set()

    dataset = WhipstrTSVSpeechDataset(
        tsv_path='data/TSV_SPEECH/speech.tsv',
        limit=10630,
        all_chars=all_chars,
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Create char to index mapping (0 is reserved for padding)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(all_chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    idx_to_char[0] = '<PAD>'
    vocab_size = len(char_to_idx) + 1  # +1 for padding
    start_token_idx = vocab_size  # Use vocab_size as start token
    
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Characters: {sorted(all_chars)}")
    
    # Save vocabulary to models/model.json
    vocab_list = sorted(all_chars)
    os.makedirs('models', exist_ok=True)
    save_vocab(vocab_list, 'models/model.json')
    print(f"   Vocabulary saved to models/model.json")
    
    # Show a sample
    sample_image, sample_solution = dataset[0]
    print(f"\n   Sample from dataset:")
    print(f"   - Spectrogram shape: {sample_image.shape}")
    print(f"   - Transcription: '{sample_solution}'")
    print(f"   - Spectrogram width: {sample_image.shape[2]} pixels")
    
    # Step 2: Create dataloaders
    print(f"\n3. Creating DataLoaders")
    
    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=32)
    
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
        vocab_size=vocab_size + 1  # +1 for start token
    ).to(device)
    print(f"   Transformer:")
    print(f"   - Model dimension: 256")
    print(f"   - Attention heads: 8")
    print(f"   - Encoder layers: 4")
    print(f"   - Decoder layers: 4")
    print(f"   - Vocabulary size: {vocab_size + 1}")
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    transformer_params = sum(p.numel() for p in transformer.parameters())
    print(f"\n   Total parameters:")
    print(f"   - Encoder: {encoder_params:,}")
    print(f"   - Transformer: {transformer_params:,}")
    print(f"   - Total: {encoder_params + transformer_params:,}")
    
    # Load finetune checkpoint if --finetune-pt was provided
    if args.finetune_pt:
        print(f"\n6. Loading finetune checkpoint from {args.finetune_pt}")
        finetune_checkpoint = torch.load(args.finetune_pt, map_location=device)
        state_dict = finetune_checkpoint["transformer_state_dict"]
        model_dict = transformer.state_dict()
        filtered_state = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        missing, unexpected = transformer.load_state_dict(filtered_state, strict=False)
        print(f"   Missing: {missing}")
        print(f"   Unexpected: {unexpected}")
        print(f"   Loaded {len(filtered_state)}/{len(state_dict)} transformer parameters")
    
    # Step 4: Setup optimizer and loss
    print(f"\n5. Setting Up Training")
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(transformer.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    print(f"   Optimizer: Adam")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Loss function: CrossEntropyLoss (ignore_index=0)")
    
    # Load checkpoint if --continue-pt was provided
    start_epoch = 0
    best_val_accuracy = -1.0

    if args.continue_pt:
        print(f"\n6. Loading checkpoint from {args.continue_pt}")
        checkpoint = torch.load(args.continue_pt, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"   Resumed from epoch {start_epoch}")

        # Evaluate to determine best_val_accuracy baseline
        print(f"   Evaluating checkpoint to establish baseline accuracy...")
        encoder.eval()
        transformer.eval()
        total_correct = 0
        total_chars = 0
        with torch.no_grad():
            for images, solution_strings, widths in tqdm(val_loader, desc="   Baseline eval"):
                images = images.to(device)
                widths = widths.to(device)
                targets = solution_string_to_tensor(solution_strings, char_to_idx, device)
                batch_size_actual, seq_len = targets.shape
                encoder_tokens = encoder(images)
                # Build encoder padding mask: True at positions beyond the sample's real windows
                T = encoder_tokens.shape[1]
                real_windows = widths - window_size + 1
                encoder_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= real_windows.unsqueeze(1)
                start_tokens = torch.full((batch_size_actual, 1), start_token_idx, dtype=torch.long, device=device)
                decoder_input = torch.cat([start_tokens, targets[:, :-1]], dim=1)
                logits = transformer(encoder_tokens, decoder_input, encoder_padding_mask=encoder_padding_mask)
                predictions = torch.argmax(logits, dim=-1)
                mask = targets != 0
                total_correct += ((predictions == targets) & mask).sum().item()
                total_chars += mask.sum().item()
        best_val_accuracy = total_correct / max(total_chars, 1)
        print(f"   Baseline val_accuracy: {best_val_accuracy:.4f}")

    # Step 5: Training loop
    step_num = 6 + (1 if args.continue_pt else 0) + (1 if args.finetune_pt else 0)
    print(f"\n{step_num}. Training for epochs {start_epoch + 1}–{num_epochs}")
    print("   " + "-" * 56)
    
    recent_epoch_dirs = []  # track the 2 most recent epoch checkpoint dirs
    global_batch_idx = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        encoder.train()
        transformer.train()
        
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, solution_strings, widths) in enumerate(tqdm(train_loader)):
            # Move to device
            images = images.to(device)
            widths = widths.to(device)
            targets = solution_string_to_tensor(solution_strings, char_to_idx, device)
            batch_size_actual, seq_len = targets.shape
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            encoder_tokens = encoder(images)
            # Build encoder padding mask: True at positions beyond the sample's real windows
            T = encoder_tokens.shape[1]
            real_windows = widths - window_size + 1
            encoder_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= real_windows.unsqueeze(1)
            
            # Prepare decoder input (shift targets, prepend start token)
            start_tokens = torch.full((batch_size_actual, 1), start_token_idx, dtype=torch.long, device=device)
            decoder_input = torch.cat([start_tokens, targets[:, :-1]], dim=1)
            
            # Transformer forward
            logits = transformer(encoder_tokens, decoder_input, encoder_padding_mask=encoder_padding_mask)
            
            # Compute loss
            logits_flat = logits.reshape(-1, vocab_size + 1)
            targets_flat = targets.reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            global_batch_idx += 1
            
            if global_batch_idx % 1000 == 0:
                ckpt_dir = f'models/checkpoint_batch_{global_batch_idx}'
                save_checkpoint(encoder, transformer, optimizer, epoch + 1,
                                0.0, ckpt_dir, vocab_list)
                print(f"   Batch checkpoint saved at {global_batch_idx} batches")
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        encoder.eval()
        transformer.eval()
        
        val_loss = 0.0
        total_correct = 0
        total_chars = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for images, solution_strings, widths in tqdm(val_loader):
                images = images.to(device)
                widths = widths.to(device)
                targets = solution_string_to_tensor(solution_strings, char_to_idx, device)
                batch_size_actual, seq_len = targets.shape
                
                # Forward pass
                encoder_tokens = encoder(images)
                # Build encoder padding mask: True at positions beyond the sample's real windows
                T = encoder_tokens.shape[1]
                real_windows = widths - window_size + 1
                encoder_padding_mask = torch.arange(T, device=device).unsqueeze(0) >= real_windows.unsqueeze(1)
                start_tokens = torch.full((batch_size_actual, 1), start_token_idx, dtype=torch.long, device=device)
                decoder_input = torch.cat([start_tokens, targets[:, :-1]], dim=1)
                logits = transformer(encoder_tokens, decoder_input, encoder_padding_mask=encoder_padding_mask)
                
                # Loss
                logits_flat = logits.reshape(-1, vocab_size + 1)
                targets_flat = targets.reshape(-1)
                loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()
                num_val_batches += 1
                
                # Accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = targets != 0
                correct = ((predictions == targets) & mask).sum().item()
                total_correct += correct
                total_chars += mask.sum().item()
        
        avg_val_loss = val_loss / max(num_val_batches, 1)
        val_accuracy = total_correct / max(total_chars, 1)
        
        print(f"   Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Acc={val_accuracy:.4f}")
        
        # --- Checkpointing ---
        # Best model by val_accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(encoder, transformer, optimizer, epoch + 1,
                            val_accuracy, 'models/best', vocab_list)
            print(f"   ✓ New best model saved (val_acc={val_accuracy:.4f})")
        
        # Most recent epoch checkpoint (keep only 2)
        epoch_dir = f'models/epoch_{epoch + 1}'
        save_checkpoint(encoder, transformer, optimizer, epoch + 1,
                        val_accuracy, epoch_dir, vocab_list)
        recent_epoch_dirs.append(epoch_dir)
        if len(recent_epoch_dirs) > 2:
            old_dir = recent_epoch_dirs.pop(0)
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
            print(f"   Removed old checkpoint: {old_dir}")
    
    print("   " + "-" * 56)
    print("   Training complete!")
    
    # Step: Inference on a test sample
    step_num = 7 + (1 if args.continue_pt else 0) + (1 if args.finetune_pt else 0)
    print(f"\n{step_num}. Inference on Test Sample")
    
    encoder.eval()
    transformer.eval()
    
    # Get a test sample from validation set
    test_idx = 0 if len(val_dataset) > 0 else 0
    test_image, test_solution = val_dataset[test_idx] if len(val_dataset) > 0 else dataset[0]
    print(f"   Ground truth: '{test_solution}'")
    
    # Prepare for inference
    test_image_batch = test_image.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        # Encode
        encoder_tokens = encoder(test_image_batch)
        print(f"   Encoder output shape: {encoder_tokens.shape}")
        
        # Generate predictions auto-regressively
        max_length = max(len(s) for _, s in dataset)
        predictions = transformer.generate(
            encoder_tokens,
            max_length=max_length,
            start_token=start_token_idx
        )
        
        # Convert to string
        predicted_indices = predictions[0].cpu().tolist()
        predicted_string = ''.join(idx_to_char.get(idx, '?') for idx in predicted_indices if idx > 0 and idx < vocab_size)
        
        print(f"   Predicted:    '{predicted_string}'")
        
        # Check if correct
        if predicted_string == test_solution:
            print(f"   ✓ Prediction matches ground truth!")
        else:
            # Count correct characters
            correct_count = sum(1 for p, t in zip(predicted_string, test_solution) if p == t)
            print(f"   Correct characters: {correct_count}/{len(test_solution)}")
    
    print(f"\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
