"""
Property-based tests for ConditioningMLP.

Feature: imagen-stage2-trainer
Property 4: Conditioning vector influence
Validates: Requirements 2.1, 2.2
"""

import torch
import pytest
from hypothesis import given, strategies as st, settings, assume
from imagen.conditioning_mlp import ConditioningMLP


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_output_shape_default():
    """ConditioningMLP with default args produces (B, out_dim) output."""
    mlp = ConditioningMLP()
    cond = torch.randn(4, 64)
    out = mlp(cond)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"


def test_output_shape_custom():
    """ConditioningMLP with custom dims produces correct output shape."""
    mlp = ConditioningMLP(in_dim=32, hidden_dim=64, out_dim=128)
    cond = torch.randn(8, 32)
    out = mlp(cond)
    assert out.shape == (8, 128), f"Expected (8, 128), got {out.shape}"


def test_two_linear_layers_with_relu():
    """ConditioningMLP has exactly fc1, relu, fc2 in the expected order."""
    mlp = ConditioningMLP(in_dim=64, hidden_dim=128, out_dim=256)
    assert isinstance(mlp.fc1, torch.nn.Linear)
    assert isinstance(mlp.relu, torch.nn.ReLU)
    assert isinstance(mlp.fc2, torch.nn.Linear)
    assert mlp.fc1.in_features == 64
    assert mlp.fc1.out_features == 128
    assert mlp.fc2.in_features == 128
    assert mlp.fc2.out_features == 256


# ---------------------------------------------------------------------------
# Property-based test — Property 4: Conditioning vector influence
# ---------------------------------------------------------------------------

# Feature: imagen-stage2-trainer, Property 4: Conditioning vector influence
# Validates: Requirements 2.1, 2.2
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    out_dim=st.integers(min_value=1, max_value=256),
)
def test_property_4_conditioning_influence(batch_size, out_dim):
    """
    Property 4: Conditioning vector influence
    For any two distinct conditioning vectors c1 != c2, the ConditioningMLP
    SHALL produce different outputs (the MLP is not a constant function).
    """
    mlp = ConditioningMLP(in_dim=64, hidden_dim=128, out_dim=out_dim)
    mlp.eval()

    # Generate two distinct conditioning vectors
    c1 = torch.randn(batch_size, 64)
    # Ensure c2 differs from c1 by adding a non-zero perturbation
    c2 = c1 + torch.randn(batch_size, 64) * 0.5 + 0.1

    with torch.no_grad():
        out1 = mlp(c1)
        out2 = mlp(c2)

    # Outputs must have the correct shape
    assert out1.shape == (batch_size, out_dim), \
        f"Expected ({batch_size}, {out_dim}), got {out1.shape}"
    assert out2.shape == (batch_size, out_dim), \
        f"Expected ({batch_size}, {out_dim}), got {out2.shape}"

    # Distinct inputs must produce distinct outputs
    assert not torch.allclose(out1, out2), \
        "ConditioningMLP produced identical outputs for distinct conditioning vectors"
