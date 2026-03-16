"""
Evaluate a Whipstr STT checkpoint on a TSV speech dataset.

Usage:
    python stt_evaluate.py --model-pt models/best/checkpoint.pt --data data/TSV_SPEECH/speech.tsv
"""

import argparse
import json

import jiwer
import jiwer.transforms as tr
import torch
from whipstr.whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
from whipstr.whipstr_encoder import WhipstrEncoder
from whipstr.whipstr_transformer import WhipstrTransformer

# Shared WER transform: lowercase, strip punctuation, normalize whitespace
wer_transform = tr.Compose([
    tr.ToLowerCase(),
    tr.RemovePunctuation(),
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    tr.ReduceToListOfListOfWords(),
])


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whipstr STT model")
    parser.add_argument('--model-pt', type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--model-json', type=str, default=None,
                        help='Path to model.json vocab file (if omitted, vocab is built from --data)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to TSV speech data file')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of samples (0 = all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    all_chars = set() if not args.model_json else None
    limit = args.limit if args.limit > 0 else 999999999
    dataset = WhipstrTSVSpeechDataset(
        tsv_path=args.data,
        limit=limit,
        all_chars=all_chars,
    )
    print(f"Dataset: {len(dataset)} samples")

    # Build vocab from model.json if provided, otherwise from dataset
    if args.model_json:
        with open(args.model_json, 'r') as f:
            vocab_list = json.load(f)["Vocab"]
        print(f"Vocabulary loaded from {args.model_json} ({len(vocab_list)} chars)")
    else:
        vocab_list = sorted(all_chars)
        print(f"Vocabulary built from dataset ({len(vocab_list)} chars)")

    char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab_list)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    idx_to_char[0] = '<PAD>'
    vocab_size = len(char_to_idx) + 1  # +1 for padding
    start_token_idx = vocab_size

    # Instantiate models (must match training architecture)
    encoder = WhipstrEncoder(stride=1, window_size=11).to(device)
    transformer = WhipstrTransformer(
        d_model=256, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=1024, dropout=0.1,
        vocab_size=vocab_size + 1,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.model_pt}")
    checkpoint = torch.load(args.model_pt, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    epoch = checkpoint.get('epoch', '?')
    print(f"Checkpoint epoch: {epoch}")

    encoder.eval()
    transformer.eval()

    # Determine max target length for generation
    max_length = max(len(s) for _, s in dataset)

    all_references = []
    all_hypotheses = []

    print(f"\n{'─' * 60}")
    with torch.no_grad():
        for i in range(len(dataset)):
            image, ground_truth = dataset[i]
            image_batch = image.unsqueeze(0).to(device)

            encoder_tokens = encoder(image_batch)
            predictions = transformer.generate(
                encoder_tokens,
                max_length=max_length
            )

            predicted_indices = predictions[0].cpu().tolist()
            predicted_text = ''.join(
                idx_to_char.get(idx, '?')
                for idx in predicted_indices
                if 0 < idx < vocab_size
            )

            all_references.append(ground_truth)
            all_hypotheses.append(predicted_text)

            sample_wer = jiwer.wer(
                ground_truth, predicted_text,
                reference_transform=wer_transform,
                hypothesis_transform=wer_transform,
            )
            print(f"[{i+1}/{len(dataset)}]  WER={sample_wer * 100:5.1f}%")
            print(f"  REF: {ground_truth}")
            print(f"  HYP: {predicted_text}")

    overall_wer = jiwer.wer(
        all_references, all_hypotheses,
        reference_transform=wer_transform,
        hypothesis_transform=wer_transform,
    )
    print(f"{'─' * 60}")
    print(f"\nOverall WER: {overall_wer * 100:.2f}%")


if __name__ == '__main__':
    main()
