"""
Imagen Stage 2: acoustic codec decoder.

Trains an ImageGenerator (U-Net) to reconstruct spectrogram windows from
64-float encoder tokens via standard DDPM ε-prediction.
"""

__version__ = "0.2.0"

from .image_generator import ImageGenerator
from .spectrogram_window_dataset import SpectrogramWindowDataset
from .imagen_train import ImagenTrainer
from .utils import validate_input_shape, validate_finite

__all__ = [
    "ImageGenerator",
    "SpectrogramWindowDataset",
    "ImagenTrainer",
    "validate_input_shape",
    "validate_finite",
]
