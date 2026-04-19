"""
Whipstr STT (ASR): A deep learning automatic speech recognition system
for transcribing speech audio into text using transformer-based models.
"""

__version__ = "0.1.0"

from .whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset
from .hf_integration import (
    WhipstrConfig,
    WhipstrTokenizer,
    WhipstrFeatureExtractor,
    WhipstrForConditionalGeneration,
    convert_to_hf,
)

__all__ = [
    'WhipstrTSVSpeechDataset',
    'WhipstrConfig',
    'WhipstrTokenizer',
    'WhipstrFeatureExtractor',
    'WhipstrForConditionalGeneration',
    'convert_to_hf',
]
