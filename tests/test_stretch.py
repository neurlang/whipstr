#!/usr/bin/env python3
"""
Verification script for Whipstr STT encoder architecture and window processing.
"""

import torch
from whipstr.whipstr_encoder import WhipstrEncoder


def test_encoder_compatibility():
    """Test that encoder handles spectrograms correctly."""
    print("=" * 70)
    print("TEST 1: Encoder Compatibility Verification")
    print("=" * 70)

    encoder = WhipstrEncoder(stride=1)

    # Test various widths (batch_size=1, small widths to keep memory low)
    test_cases = [
        (28, 1),   # Single window
        (40, 13),  # Small
        (56, 29),  # Two segments
    ]

    for width, expected_frames in test_cases:
        test_image = torch.rand(1, 2, 836, width)
        output = encoder(test_image)

        assert output.shape == (1, expected_frames, 10), \
            f"Expected shape (1, {expected_frames}, 10), got {output.shape}"

        # Check output is finite
        assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf"

        print(f"✓ Width {width:3d} -> {expected_frames:3d} frames (shape: {output.shape})")

    print(f"✓ Encoder processes all widths correctly")
    print()


def test_encoder_architecture():
    """Test encoder architecture details."""
    print("=" * 70)
    print("TEST 2: Encoder Architecture Verification")
    print("=" * 70)

    encoder = WhipstrEncoder(stride=1)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Check architecture components
    assert hasattr(encoder, 'conv1'), "Missing conv1 layer"
    assert hasattr(encoder, 'conv2'), "Missing conv2 layer"
    assert hasattr(encoder, 'conv3'), "Missing conv3 layer"
    assert hasattr(encoder, 'conv4'), "Missing conv4 layer"
    assert hasattr(encoder, 'fc1'), "Missing fc1 layer"
    assert hasattr(encoder, 'fc2'), "Missing fc2 layer"

    print(f"✓ All expected layers present")

    # Check layer dimensions
    assert encoder.conv1.in_channels == 2, "Conv1 should have 2 input channels"
    assert encoder.conv1.out_channels == 32, "Conv1 should have 32 output channels"
    assert encoder.conv4.out_channels == 256, "Conv4 should have 256 output channels"
    assert encoder.fc2.out_features == 10, "FC2 should output 10 classes"

    print(f"✓ Layer dimensions correct")
    print()


def test_window_extraction():
    """Test that windows are extracted correctly from spectrograms."""
    print("=" * 70)
    print("TEST 3: Window Extraction Verification")
    print("=" * 70)

    # Create a test image with known pattern
    test_image = torch.zeros(1, 2, 836, 42)

    # Put a marker in a specific window location
    test_image[0, 0, 10:38, 28:56] = 1.0

    encoder = WhipstrEncoder(stride=1)

    # Extract windows manually to verify
    windows = test_image.unfold(3, 28, 1)  # [1, 2, 56, num_windows, 28]
    num_windows = windows.shape[3]

    print(f"✓ Image shape: {test_image.shape}")
    print(f"✓ Number of windows: {num_windows}")
    print(f"✓ Windows shape: {windows.shape}")

    # Verify window dimensions
    assert windows.shape[2] == 836, f"Window height should be 836, got {windows.shape[2]}"
    assert windows.shape[4] == 28, f"Window width should be 28, got {windows.shape[4]}"

    # Process through encoder
    output = encoder(test_image)
    assert output.shape == (1, num_windows, 10), f"Expected shape (1, {num_windows}, 10), got {output.shape}"

    print(f"✓ Windows extracted correctly with shape [2, 836, 28]")
    print(f"✓ Encoder output shape: {output.shape}")
    print()


def main():
    """Run all verification tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "WHIPSTR STT ENCODER VERIFICATION" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_encoder_compatibility()
        test_encoder_architecture()
        test_window_extraction()

        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print()
        print("=" * 70)
        print("✗ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        return 1

    except Exception as e:
        print()
        print("=" * 70)
        print("✗ UNEXPECTED ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
