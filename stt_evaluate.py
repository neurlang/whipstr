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
from whipstr.whipstr_variants import get_variant_config, list_variants

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
    parser.add_argument('--force-cpu', type=bool, default=False,
                        help='Force cpu (default False)')
    parser.add_argument('--variant', type=str, default='whipstr-base',
                        choices=list_variants(),
                        help='Model variant (default: whipstr-base)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
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

    char_to_idx = {char: i + 1 for i, char in enumerate(vocab_list)}
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    pad_id = 0
    eos_id = len(char_to_idx) + 1
    transformer_vocab_size = eos_id + 1  # 0 + N chars + 1 for EOS

    # Instantiate models from variant config
    cfg = get_variant_config(args.variant, vocab_size=transformer_vocab_size)
    encoder = WhipstrEncoder(
        stride=cfg["stride"], window_size=cfg["window_size"],
        output_values=cfg["encoder_embed_dim"],
    ).to(device)
    transformer = WhipstrTransformer(
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        vocab_size=cfg["vocab_size"],
        input_values=cfg["encoder_embed_dim"],
    ).to(device)
    print(f"Variant: {args.variant}")
    print(f"  encoder_embed_dim={cfg['encoder_embed_dim']}, d_model={cfg['d_model']}, "
          f"nhead={cfg['nhead']}, layers={cfg['num_encoder_layers']}/{cfg['num_decoder_layers']}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.model_pt}")
    checkpoint = torch.load(args.model_pt, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    epoch = checkpoint.get('epoch', '?')
    # Use checkpoint's eos_id if available, fall back to computed
    eos_id = checkpoint.get('eos_id', eos_id)
    print(f"Checkpoint epoch: {epoch}, eos_id: {eos_id}")

    encoder.eval()
    transformer.eval()

    all_references = []
    all_hypotheses = []

    print(f"\n{'─' * 60}")
    with torch.inference_mode():
        for i in range(len(dataset)):
            image, ground_truth = dataset[i]
            image_batch = image.unsqueeze(0).to(device)

            encoder_tokens = encoder(image_batch)

            del image_batch
            encoder_tokens = encoder_tokens.cpu()

            predictions = transformer.generate(
                encoder_tokens.to(device),
                max_length = len(ground_truth) + 1,
                start_token=pad_id,
                eos_token=eos_id,
            )

            del encoder_tokens

            predicted_indices = predictions[0].cpu().tolist()

            del predictions

            # Stop at first EOS token
            if eos_id in predicted_indices:
                predicted_indices = predicted_indices[:predicted_indices.index(eos_id)]

            predicted_text = ''.join(
                idx_to_char.get(idx, '?')
                for idx in predicted_indices
                if 0 < idx < eos_id
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
