"""
Imagen Stage 2 Trainer: trains an asymmetric 2D U-Net (ImageGenerator) as the
encoder half of an autoencoder whose decoder is the frozen Stage 1 WhipstrEncoder.
"""

__version__ = "0.1.0"

from .image_generator import ImageGenerator
from .conditioning_mlp import ConditioningMLP
from .spectrogram_window_dataset import SpectrogramWindowDataset
from .imagen_train import ImagenTrainer
from .utils import validate_input_shape, validate_finite

__all__ = [
    "ImageGenerator",
    "ConditioningMLP",
    "SpectrogramWindowDataset",
    "ImagenTrainer",
    "validate_input_shape",
    "validate_finite",
]
