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


def test_unit_encoder_frozen_after_load():
    """All encoder parameters must have requires_grad=False after loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        for name, param in trainer.encoder.named_parameters():
            assert not param.requires_grad, (
                f"Encoder parameter '{name}' should be frozen (requires_grad=False)"
            )


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


def test_unit_load_checkpoint_restores_epoch():
    """load_checkpoint restores current_epoch to the saved epoch value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)
        # Manually save a checkpoint at epoch 7
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        torch.save(
            {
                "epoch": 7,
                "generator_state_dict": trainer.generator.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "loss_noise": 0.1,
                "loss_recon": 0.05,
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
# Property 5: Frozen encoder invariance
# Feature: imagen-stage2-trainer, Property 5: Frozen encoder invariance
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=2))
def test_property_5_frozen_encoder_invariance(batch_size):
    """
    Property 5: Frozen encoder invariance
    For any input passed through the frozen WhipstrEncoder before and after
    Stage 2 training begins, the encoder SHALL produce identical outputs
    (weights unchanged).

    # Feature: imagen-stage2-trainer, Property 5: Frozen encoder invariance
    # Validates: Requirements 3.2
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)

        # WhipstrEncoder expects (B, 2, 836, W); use W=11 (minimum window_size)
        x = torch.randn(batch_size, 2, 836, 11)

        # Capture encoder output before any training step
        with torch.no_grad():
            out_before = trainer.encoder(x)

        # Simulate a training step on the generator (encoder must stay frozen)
        windows = torch.randn(batch_size, 2, 11, 836)
        cond = torch.randn(batch_size, 64)
        trainer.generator.train()
        trainer.optimizer.zero_grad()
        loss, _, _ = trainer.compute_loss(windows, cond)
        loss.backward()
        trainer.optimizer.step()

        # Encoder output must be identical after the training step
        with torch.no_grad():
            out_after = trainer.encoder(x)

        assert torch.equal(out_before, out_after), (
            "Frozen encoder produced different outputs after a generator training step — "
            "encoder weights were modified."
        )


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

    # Feature: imagen-stage2-trainer, Property 8: Optimizer state round-trip
    # Validates: Requirements 7.2
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_trainer(tmpdir)

        # Do a training step so the optimizer has non-trivial state
        windows = torch.randn(batch_size, 2, 11, 836)
        cond = torch.randn(batch_size, 64)
        trainer.generator.train()
        trainer.optimizer.zero_grad()
        loss, _, _ = trainer.compute_loss(windows, cond)
        loss.backward()
        trainer.optimizer.step()

        # Manually set epoch to a known value
        trainer.current_epoch = 3

        # Save checkpoint
        ckpt_path = trainer._save_checkpoint(
            epoch=trainer.current_epoch,
            loss_noise=0.1,
            loss_recon=0.05,
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

        # Compare param_groups (lr, betas, etc.)
        for key in saved_opt_state["param_groups"][0]:
            assert saved_opt_state["param_groups"][0][key] == reloaded_opt_state["param_groups"][0][key], (
                f"Optimizer param_group key '{key}' mismatch after round-trip"
            )

        # Compare per-parameter state tensors
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
