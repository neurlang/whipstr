"""
Property-based and unit tests for ImagenTrainer.

Feature: imagen-stage2-trainer
Validates: Requirements 3.2, 7.2
"""

import os
import tempfile
import torch
import pytest
from hypothesis import given, strategies as st, settings

from whipstr.whipstr_encoder import WhipstrEncoder
from imagen.image_generator import ImageGenerator
from imagen.imagen_train import ImagenTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_encoder_checkpoint(path: str) -> None:
    """Save a fresh WhipstrEncoder state_dict to a temp checkpoint file."""
    encoder = WhipstrEncoder(window_size=11)
    torch.save(encoder.state_dict(), path)


def _make_trainer(tmpdir: str) -> ImagenTrainer:
    """Create an ImagenTrainer with a fresh encoder checkpoint."""
    ckpt_path = os.path.join(tmpdir, "encoder.pt")
    _save_encoder_checkpoint(ckpt_path)
    return ImagenTrainer(
        encoder_checkpoint_path=ckpt_path,
        generator=ImageGenerator(),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_unit_missing_encoder_checkpoint_raises():
    """FileNotFoundError when encoder checkpoint path does not exist."""
    with pytest.raises(FileNotFoundError):
        ImagenTrainer(encoder_checkpoint_path="/nonexistent/path.pt")


def test_unit_add_noise_shape():
    """add_noise returns (x_t, noise) both with same shape as x_clean."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        x = torch.randn(2, 2, 11, 836)
        t = torch.rand(2)
        x_t, noise = trainer.add_noise(x, t)
        assert x_t.shape == x.shape
        assert noise.shape == x.shape


def test_unit_add_noise_formula():
    """x_t = x_clean + t * noise (deterministic check with fixed seed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        torch.manual_seed(0)
        x = torch.randn(1, 2, 11, 836)
        t = torch.tensor([0.5])
        torch.manual_seed(42)
        x_t, noise = trainer.add_noise(x, t)
        torch.manual_seed(42)
        expected_noise = torch.randn_like(x)
        assert torch.allclose(x_t, x + 0.5 * expected_noise)


def test_unit_compute_loss_returns_scalar():
    """compute_loss returns a single scalar tensor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        windows = torch.randn(2, 2, 11, 836)
        cond = torch.randn(2, 64)
        loss = trainer.compute_loss(windows, cond)
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


def test_unit_compute_loss_backward():
    """Loss is differentiable through the generator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        windows = torch.randn(1, 2, 11, 836)
        cond = torch.randn(1, 64)
        loss = trainer.compute_loss(windows, cond)
        loss.backward()
        # All generator parameters should have gradients
        for name, param in trainer.generator.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient after loss.backward()"
            )


def test_unit_load_checkpoint_restores_epoch():
    """load_checkpoint restores current_epoch to the saved epoch value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        torch.save(
            {
                "epoch": 7,
                "generator_state_dict": trainer.generator.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "loss": 0.1,
                "config": trainer.generator.get_config(),
            },
            ckpt_path,
        )
        trainer.load_checkpoint(ckpt_path)
        assert trainer.current_epoch == 7


def test_unit_load_checkpoint_missing_raises():
    """FileNotFoundError when checkpoint path does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("/nonexistent/ckpt.pt")


# ---------------------------------------------------------------------------
# Property 8: Optimizer state round-trip
# Feature: imagen-stage2-trainer, Property 8: Optimizer state round-trip
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=2))
def test_property_8_optimizer_state_roundtrip(batch_size):
    """
    Property 8: Optimizer state round-trip
    For any trainer checkpoint, saving and reloading SHALL restore the
    optimizer state dict and epoch counter to their exact saved values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)

        # Do a training step so the optimizer has non-trivial state
        windows = torch.randn(batch_size, 2, 11, 836)
        cond = torch.randn(batch_size, 64)
        trainer.generator.train()
        trainer.optimizer.zero_grad()
        loss = trainer.compute_loss(windows, cond)
        loss.backward()
        trainer.optimizer.step()

        # Manually set epoch to a known value
        trainer.current_epoch = 3

        # Save checkpoint
        ckpt_path = trainer._save_checkpoint(
            epoch=trainer.current_epoch,
            loss=loss.item(),
            output_dir=tmpdir,
        )

        # Capture state before reload
        saved_opt_state = trainer.optimizer.state_dict()
        saved_epoch = trainer.current_epoch

        # Create a fresh trainer and reload
        trainer2 = _make_trainer(tmpdir)
        trainer2.load_checkpoint(ckpt_path)

        # Epoch must match exactly
        assert trainer2.current_epoch == saved_epoch, (
            f"Epoch mismatch: expected {saved_epoch}, got {trainer2.current_epoch}"
        )

        # Optimizer state must match exactly
        reloaded_opt_state = trainer2.optimizer.state_dict()

        for key in saved_opt_state["param_groups"][0]:
            assert saved_opt_state["param_groups"][0][key] == reloaded_opt_state["param_groups"][0][key], (
                f"Optimizer param_group key '{key}' mismatch after round-trip"
            )

        for param_id, saved_p_state in saved_opt_state["state"].items():
            reloaded_p_state = reloaded_opt_state["state"][param_id]
            for k, v in saved_p_state.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, reloaded_p_state[k]), (
                        f"Optimizer state tensor '{k}' for param {param_id} "
                        "differs after round-trip"
                    )
                else:
                    assert v == reloaded_p_state[k], (
                        f"Optimizer state value '{k}' for param {param_id} "
                        "differs after round-trip"
                    )
