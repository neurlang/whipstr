"""
Property-based tests for SpectrogramWindowDataset.

Feature: imagen-stage2-trainer
Validates: Requirements 6.2, 6.3, 6.4
"""

import torch
import pytest
from hypothesis import given, strategies as st, settings

from imagen.spectrogram_window_dataset import SpectrogramWindowDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(W: int) -> torch.Tensor:
    """Create a synthetic (2, 836, W) spectrogram tensor."""
    return torch.randn(2, 836, W)


# ---------------------------------------------------------------------------
# Property 9: Dataset window shape
# Feature: imagen-stage2-trainer, Property 9: Dataset window shape
# Validates: Requirements 6.2, 6.4
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(W=st.integers(min_value=11, max_value=220))
def test_property_9_dataset_window_shape(W):
    """
    Property 9: Dataset window shape
    For any spectrogram of shape (2, 836, W) where W >= 11, the
    SpectrogramWindowDataset SHALL yield windows of shape (2, 11, 836)
    and conditioning vectors of shape (64,).
    """
    spec = _make_spec(W)
    ds = SpectrogramWindowDataset([spec])

    for i in range(len(ds)):
        window, cond = ds[i]
        assert window.shape == (2, 11, 836), (
            f"Expected window shape (2, 11, 836), got {tuple(window.shape)}"
        )
        assert cond.shape == (64,), (
            f"Expected cond shape (64,), got {tuple(cond.shape)}"
        )


# ---------------------------------------------------------------------------
# Property 10: Dataset window count
# Feature: imagen-stage2-trainer, Property 10: Dataset window count
# Validates: Requirements 6.2, 6.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(W=st.integers(min_value=11, max_value=220))
def test_property_10_dataset_window_count(W):
    """
    Property 10: Dataset window count
    For any spectrogram of width W, the number of windows yielded SHALL
    equal W // 11.
    """
    spec = _make_spec(W)
    ds = SpectrogramWindowDataset([spec])

    expected = W // 11
    assert len(ds) == expected, (
        f"Expected {expected} windows for W={W}, got {len(ds)}"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_unit_invalid_source_type():
    with pytest.raises(TypeError):
        SpectrogramWindowDataset(42)


def test_unit_invalid_spec_shape():
    bad = torch.randn(2, 100, 50)  # wrong freq dim
    with pytest.raises(ValueError):
        SpectrogramWindowDataset([bad])


def test_unit_width_too_small():
    tiny = torch.randn(2, 836, 5)  # W < 11
    with pytest.raises(ValueError):
        SpectrogramWindowDataset([tiny])


def test_unit_cond_position_norm():
    """First window position_norm should be 0.0, last should be 1.0 (for n>1)."""
    W = 33  # 3 windows
    spec = _make_spec(W)
    ds = SpectrogramWindowDataset([spec])
    assert len(ds) == 3

    _, cond_first = ds[0]
    _, cond_last  = ds[2]
    assert cond_first[0].item() == pytest.approx(0.0)
    assert cond_last[0].item()  == pytest.approx(1.0)


def test_unit_multiple_spectrograms():
    """Dataset correctly concatenates windows from multiple spectrograms."""
    specs = [_make_spec(22), _make_spec(33)]  # 2 + 3 = 5 windows
    ds = SpectrogramWindowDataset(specs)
    assert len(ds) == 5
