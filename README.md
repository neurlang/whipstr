# Whipstr STT (ASR)

A deep learning automatic speech recognition (ASR) system for transcribing speech audio into text using transformer-based sequence-to-sequence models.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
whipstr/
├── whipstr_tsv_speech_dataset.py # TSV speech dataset loader
├── whipstr_encoder.py            # CNN audio encoder
├── whipstr_transformer.py        # Transformer seq-to-seq model
└── whipstr_train.py              # Training pipeline

tests/
└── (test files)

requirements.txt                  # Dependencies
```

## Usage

See `example.py` for usage examples.

## Training

```bash
# Quick training example
python example.py

# Full training with improved hyperparameters
python train_improved.py
```

## Documentation

- [Changes Summary](doc/CHANGES_SUMMARY.txt) — Spectrogram height changes
- [Fixes Summary](doc/FIXES_SUMMARY.md) — Training issue fixes
- [Spectrogram Height Changes](doc/IMAGE_STRETCH_CHANGES.md) — Technical details on spectrogram resize
- [Spectrogram Height Summary](doc/IMAGE_STRETCH_SUMMARY.md) — Implementation summary
- [Quick Start (56px)](doc/QUICK_START_56PX.md) — Quick start guide
