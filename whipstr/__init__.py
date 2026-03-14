"""
Whipstr STT (ASR): A deep learning automatic speech recognition system
for transcribing speech audio into text using transformer-based models.
"""

__version__ = "0.1.0"

from .whipstr_tsv_speech_dataset import WhipstrTSVSpeechDataset

__all__ = ['WhipstrTSVSpeechDataset']
