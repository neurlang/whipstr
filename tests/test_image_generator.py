"""
Property-based and unit tests for ImageGenerator.

Feature: imagen-stage2-trainer
Validates: Requirements 1.1–1.6, 2.1–2.2, 7.1, 7.3
"""

import io
import tempfile
import os
import torch
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from imagen.image_generator import ImageGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model() -> ImageGenerator:
    return ImageGenerator()


def _make_inputs(batch_size: int):
    x = torch.randn(batch_size, 2, 11, 836)
    cond = torch.randn(batch_size, 64)
    return x, cond


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_unit_output_shape():
    model = _make_model()
    x, cond = _make_inputs(2)
    with torch.no_grad():
        out = model(x, cond)
    assert out.shape == (2, 2, 11, 836)


def test_unit_get_config():
    model = ImageGenerator(in_channels=2, base_channels=64, cond_dim=64, cond_hidden=128)
    cfg = model.get_config()
    assert cfg == {"in_channels": 2, "base_channels": 64, "cond_dim": 64, "cond_hidden": 128}


def test_unit_reconstruct_from_config():
    model = _make_model()
    cfg = model.get_config()
    model2 = ImageGenerator(**cfg)
    assert model2.get_config() == cfg


def test_unit_invalid_input_shape():
    model = _make_model()
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 10, 836), torch.randn(1, 64))


def test_unit_nan_input_raises():
    model = _make_model()
    x = torch.randn(1, 2, 11, 836)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(x, torch.randn(1, 64))


# ---------------------------------------------------------------------------
# Property 2: Time axis collapse and restore
# Feature: imagen-stage2-trainer, Property 2: Time axis collapse and restore
# Validates: Requirements 1.2, 1.5
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=4))
def test_property_2_time_axis_collapse_and_restore(batch_size):
    """
    Property 2: Time axis collapse and restore
    For any valid input, the intermediate representation after time_down SHALL
    have time dimension 1, and the final output SHALL have time dimension 11.
    """
    model = _make_model()
    model.eval()
    x, cond = _make_inputs(batch_size)

    with torch.no_grad():
        # Check intermediate: time_down collapses 11 → 1
        after_time_down = model.time_down(x)
        assert after_time_down.shape[2] == 1, (
            f"Expected time dim 1 after time_down, got {after_time_down.shape[2]}"
        )

        # Check final output restores 1 → 11
        out = model(x, cond)
        assert out.shape[2] == 11, (
            f"Expected time dim 11 in output, got {out.shape[2]}"
        )


# ---------------------------------------------------------------------------
# Property 3: Frequency bottleneck size
# Feature: imagen-stage2-trainer, Property 3: Frequency bottleneck size
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=4))
def test_property_3_frequency_bottleneck_size(batch_size):
    """
    Property 3: Frequency bottleneck size
    For any valid input, the bottleneck feature map SHALL have a frequency
    dimension <= 105.
    """
    model = _make_model()
    model.eval()
    x, cond = _make_inputs(batch_size)

    with torch.no_grad():
        # Trace through encoder to bottleneck
        s0 = torch.relu(model.time_down(x))
        s1 = torch.relu(model.freq_down1(s0))
        s2 = torch.relu(model.freq_down2(s1))
        s3 = torch.relu(model.freq_down3(s2))

    freq_dim = s3.shape[3]
    assert freq_dim <= 105, (
        f"Bottleneck frequency dimension {freq_dim} exceeds 105"
    )


# ---------------------------------------------------------------------------
# Property 1: Output shape identity
# Feature: imagen-stage2-trainer, Property 1: Output shape identity
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=4))
def test_property_1_output_shape_identity(batch_size):
    """
    Property 1: Output shape identity
    For any batch of spectrogram windows of shape (B, 2, 11, 836) and any
    conditioning vector of shape (B, 64), the ImageGenerator output SHALL
    have the same shape (B, 2, 11, 836).
    """
    model = _make_model()
    model.eval()
    x, cond = _make_inputs(batch_size)

    with torch.no_grad():
        out = model(x, cond)

    assert out.shape == (batch_size, 2, 11, 836), (
        f"Expected ({batch_size}, 2, 11, 836), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Property 6: Noise prediction finite values
# Feature: imagen-stage2-trainer, Property 6: Noise prediction finite values
# Validates: Requirements 1.1, 4.2
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=4))
def test_property_6_finite_output_values(batch_size):
    """
    Property 6: Noise prediction finite values
    For any finite input window and conditioning vector, the ImageGenerator
    output SHALL contain only finite values (no NaN or Inf).
    """
    model = _make_model()
    model.eval()
    x, cond = _make_inputs(batch_size)

    with torch.no_grad():
        out = model(x, cond)

    assert torch.isfinite(out).all(), (
        "ImageGenerator output contains NaN or Inf values"
    )


# ---------------------------------------------------------------------------
# Property 4: Conditioning vector influence
# Feature: imagen-stage2-trainer, Property 4: Conditioning vector influence
# Validates: Requirements 2.1, 2.2
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=4))
def test_property_4_conditioning_influence(batch_size):
    """
    Property 4: Conditioning vector influence
    For any two distinct conditioning vectors c1 != c2 applied to the same
    noisy input, the ImageGenerator SHALL produce different outputs.
    """
    model = _make_model()
    model.eval()
    x, c1 = _make_inputs(batch_size)
    # Ensure c2 is meaningfully different from c1
    c2 = c1 + torch.randn_like(c1) * 0.5 + 0.1

    with torch.no_grad():
        out1 = model(x, c1)
        out2 = model(x, c2)

    assert not torch.allclose(out1, out2), (
        "ImageGenerator produced identical outputs for distinct conditioning vectors"
    )


# ---------------------------------------------------------------------------
# Property 7: Checkpoint round-trip
# Feature: imagen-stage2-trainer, Property 7: Checkpoint round-trip
# Validates: Requirements 7.1, 7.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=2))
def test_property_7_checkpoint_roundtrip(batch_size):
    """
    Property 7: Checkpoint round-trip
    For any ImageGenerator model, saving with torch.save and reloading with
    torch.load SHALL produce a model whose state_dict is identical to the
    original.
    """
    model = _make_model()
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        torch.save({"config": model.get_config(), "state_dict": model.state_dict()}, path)

        ckpt = torch.load(path, weights_only=False)
        model2 = ImageGenerator(**ckpt["config"])
        model2.load_state_dict(ckpt["state_dict"])
        model2.eval()

    sd1 = model.state_dict()
    sd2 = model2.state_dict()
    assert sd1.keys() == sd2.keys(), "State dict keys differ after round-trip"
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), (
            f"State dict mismatch at key '{key}' after round-trip"
        )
