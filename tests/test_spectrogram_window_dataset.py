"""
Property-based and unit tests for SpectrogramWindowDataset.

Feature: imagen-stage2-trainer
Validates: Requirements 6.2, 6.3, 6.4
"""

import os
import tempfile
import torch
import pytest
from hypothesis import given, strategies as st, settings

from whipstr.whipstr_encoder import WhipstrEncoder
from imagen.spectrogram_window_dataset import SpectrogramWindowDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_encoder_checkpoint(path: str) -> None:
    encoder = WhipstrEncoder(window_size=11)
    torch.save(encoder.state_dict(), path)


def _make_spec(W: int) -> torch.Tensor:
    """Create a synthetic (2, 836, W) spectrogram tensor."""
    return torch.randn(2, 836, W)


def _make_dataset(specs, tmpdir):
    """Create a SpectrogramWindowDataset with a temp encoder checkpoint."""
    ckpt_path = os.path.join(tmpdir, "encoder.pt")
    _save_encoder_checkpoint(ckpt_path)
    return SpectrogramWindowDataset(
        source=specs,
        encoder_checkpoint_path=ckpt_path,
        device="cpu",
    )


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
    SpectrogramWindowDataset SHALL yield windows of shape (2, 836, 11)
    and conditioning tokens of shape (64,).
    """
    spec = _make_spec(W)
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_dataset([spec], tmpdir)

        for i in range(len(ds)):
            window, token = ds[i]
            assert window.shape == (2, 836, 11), (
                f"Expected window shape (2, 836, 11), got {tuple(window.shape)}"
            )
            assert token.shape == (64,), (
                f"Expected token shape (64,), got {tuple(token.shape)}"
            )


# ---------------------------------------------------------------------------
# Property 10: Dataset window count (overlapping)
# Feature: imagen-stage2-trainer, Property 10: Dataset window count
# Validates: Requirements 6.2, 6.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(W=st.integers(min_value=11, max_value=220))
def test_property_10_dataset_window_count(W):
    """
    Property 10: Dataset window count (overlapping)
    For any spectrogram of width W, the number of windows yielded SHALL
    equal W - 11 + 1 (overlapping windows with stride=1).
    """
    spec = _make_spec(W)
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_dataset([spec], tmpdir)

        expected = W - 11 + 1
        assert len(ds) == expected, (
            f"Expected {expected} windows for W={W}, got {len(ds)}"
        )


# ---------------------------------------------------------------------------
# Property: Token correctness (encoder output matches)
# Feature: imagen-stage2-trainer
# ---------------------------------------------------------------------------

def test_property_token_matches_encoder():
    """
    For any window, the token SHALL equal the frozen encoder's output
    when run on that window.
    """
    W = 33  # produces 23 overlapping windows
    spec = _make_spec(W)
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_dataset([spec], tmpdir)

        for i in range(len(ds)):
            window, token = ds[i]
            # window is (2, 836, 11); unsqueeze for encoder
            window_enc = window.unsqueeze(0)  # (1, 2, 836, 11)
            with torch.no_grad():
                expected_token = ds.encoder(window_enc)  # (1, 1, 64)
            expected_token = expected_token.squeeze(0).squeeze(0)  # (64,)

            assert torch.allclose(token, expected_token, atol=1e-6, rtol=1e-4), (
                f"Token at index {i} does not match encoder output"
            )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_unit_invalid_source_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "encoder.pt")
        _save_encoder_checkpoint(ckpt_path)
        with pytest.raises(TypeError):
            SpectrogramWindowDataset(42, encoder_checkpoint_path=ckpt_path)


def test_unit_invalid_spec_shape():
    bad = torch.randn(2, 100, 50)  # wrong freq dim
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "encoder.pt")
        _save_encoder_checkpoint(ckpt_path)
        with pytest.raises(ValueError):
            SpectrogramWindowDataset([bad], encoder_checkpoint_path=ckpt_path)


def test_unit_width_too_small():
    tiny = torch.randn(2, 836, 5)  # W < 11
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "encoder.pt")
        _save_encoder_checkpoint(ckpt_path)
        with pytest.raises(ValueError):
            SpectrogramWindowDataset([tiny], encoder_checkpoint_path=ckpt_path)


def test_unit_multiple_spectrograms():
    """Dataset correctly concatenates windows from multiple spectrograms."""
    specs = [_make_spec(22), _make_spec(33)]
    # W=22 → 12 windows, W=33 → 23 windows → total 35
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_dataset(specs, tmpdir)
        assert len(ds) == 35


def test_unit_token_is_finite():
    """Tokens must contain only finite values."""
    W = 33
    spec = _make_spec(W)
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_dataset([spec], tmpdir)
        for i in range(len(ds)):
            _, token = ds[i]
            assert torch.isfinite(token).all(), (
                f"Token at index {i} contains NaN or Inf"
            )
