# Whipstr STT (ASR)

A deep learning automatic speech recognition (ASR) system for transcribing speech audio into text using transformer-based sequence-to-sequence models.

| Architecture |
|--------------|
| <img width="1298" height="1212" alt="image" src="https://github.com/user-attachments/assets/9db9c2f4-8aa2-4e9d-a317-935c8b83b8b8" /> |

## Installation

Core dependencies (required):

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install whipstr[hf]      # Hugging Face integration (transformers, numpy, scipy, soundfile, sounddevice)
pip install whipstr[train]   # Training (tqdm)
pip install whipstr[dev]     # Development (hypothesis, pytest)
```

## Project Structure

```
whipstr/                       # Core library package
├── __init__.py                # Package exports (dataset, native config)
├── whipstr_config.py          # Native config dataclass (no HF dependency)
├── whipstr_encoder.py         # CNN audio encoder
├── whipstr_transformer.py     # Transformer seq-to-seq model
├── whipstr_train.py           # Training pipeline
├── whipstr_tsv_speech_dataset.py # TSV speech dataset loader
├── whipstr_variants.py        # Model variant configs (small/base/medium/large)
└── hf_integration.py          # HuggingFace integration (requires whipstr[hf])

scripts/
├── stt_example.py             # Training with variant support, checkpointing, WER eval
├── stt_evaluate.py            # Evaluate a checkpoint with WER
├── stt_infer_hf.py            # Inference via HuggingFace Hub model
└── stt_mic_hf.py              # Microphone-based inference via HF model

tests/                         # Property-based and unit tests
└── ...

pyproject.toml                 # Project metadata and optional dependencies
requirements.txt               # Core dependencies
```

## Variants

Four model sizes are configurable via `--variant`:

| Variant | Encoder dim | d_model | Heads | Enc/Dec layers | Feedforward | **Params** |
|---|---|---|---|---|---|---|
| whipstr-small | 32 | 128 | 4 | 2/2 | 512 | 8M |
| whipstr-base | 64 | 256 | 8 | 4/4 | 1024 | 17M |
| whipstr-medium | 128 | 512 | 8 | 6/6 | 2048 | 51M |
| whipstr-large | 192 | 768 | 12 | 6/6 | 3072 | 106M |

## Usage

```bash
# Train a model
python stt_example.py --variant whipstr-base

# Evaluate a checkpoint
python stt_evaluate.py --model-pt checkpoints/best.pt --data data/TSV_SPEECH/speech.tsv

# Inference via HuggingFace Hub
python stt_infer_hf.py --audio audio.wav --model ./hf_whipstr
```

## HuggingFace Integration

Convert checkpoints to HF format and upload:

```bash
# Install with HF extras first
pip install whipstr[hf]

# Then convert
python -m whipstr.hf_integration \
  --checkpoint checkpoints/best_model.pt --model-json models/model.json
```

Then use with `pipeline`:

```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition",
                model="neuralang/en-whipstr-base-48khz-libritts-r",
                trust_remote_code=True)
transcription = pipe("audio.wav")
```
