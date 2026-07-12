import torch
from hypothesis import given, strategies as st, settings
from whipstr.whipstr_encoder import WhipstrEncoder
import pytest


# Property 7: Variable width handling
# Feature: whipstr-stt-asr, Property 7: Variable width handling
# Validates: Requirements 3.1
@settings(max_examples=5, deadline=None)
@given(
    width=st.integers(min_value=28, max_value=56),
    stride=st.integers(min_value=5, max_value=10)
)
def test_property_7_variable_width_handling(width, stride):
    """
    Property 7: Variable width handling
    For any valid spectrogram tensor of shape [2, 836, W] where W >= 28, 
    the CNN encoder should process it without errors and produce output.
    """
    batch_size = 1
    encoder = WhipstrEncoder(stride=stride)
    
    # Create random input image
    image = torch.rand(batch_size, 2, 836, width)
    
    # Should process without errors
    try:
        output = encoder(image)
        
        # Output should be a tensor
        assert isinstance(output, torch.Tensor), \
            f"Output should be a tensor, got {type(output)}"
        
        # Output should have correct batch size
        assert output.shape[0] == batch_size, \
            f"Expected batch size {batch_size}, got {output.shape[0]}"
        
        # Output should have 64 values per token
        assert output.shape[2] == 64, \
            f"Expected 64 values per token, got {output.shape[2]}"
        
    except Exception as e:
        pytest.fail(f"Encoder failed to process width {width}: {e}")


# Property 8: Frame count formula
# Feature: whipstr-stt-asr, Property 8: Frame count formula
# Validates: Requirements 3.2
@settings(max_examples=5, deadline=None)
@given(
    width=st.integers(min_value=28, max_value=56),
    stride=st.integers(min_value=5, max_value=10)
)
def test_property_8_frame_count_formula(width, stride):
    """
    Property 8: Frame count formula
    For any spectrogram of width W and stride S, the number of output tokens T 
    should equal (W - 28) // S + 1.
    """
    encoder = WhipstrEncoder(stride=stride)
    
    # Create random input image
    image = torch.rand(1, 2, 836, width)
    
    # Process through encoder
    output = encoder(image)
    
    # Calculate expected number of frames
    expected_frames = (width - 28) // stride + 1
    actual_frames = output.shape[1]
    
    assert actual_frames == expected_frames, \
        f"Expected {expected_frames} frames, got {actual_frames} " \
        f"(width={width}, stride={stride})"


# Property 9: Valid token values
# Feature: whipstr-stt-asr, Property 9: Valid token values
# Validates: Requirements 3.3, 3.4
@settings(max_examples=5, deadline=None)
@given(
    width=st.integers(min_value=28, max_value=56),
    stride=st.integers(min_value=5, max_value=10)
)
def test_property_9_valid_token_values(width, stride):
    """
    Property 9: Valid token values
    For any frame processed by the CNN encoder, the output token should 
    contain exactly 64 float values (raw logits, no range restriction).
    """
    batch_size = 1
    encoder = WhipstrEncoder(stride=stride)
    
    # Create random input image
    image = torch.rand(batch_size, 2, 836, width)
    
    # Process through encoder
    output = encoder(image)
    
    # Check that each token has exactly 64 values
    assert output.shape[2] == 64, \
        f"Expected 64 values per token, got {output.shape[2]}"
    
    # Check that values are finite (no NaN or Inf)
    assert torch.all(torch.isfinite(output)), \
        "Found non-finite values (NaN or Inf) in output"
    
    # Note: No range restriction for raw logits - they can be any finite value


# Unit Tests for edge cases

def test_minimum_width_single_frame():
    """Test with minimum width (W=28, single frame)."""
    encoder = WhipstrEncoder(stride=1)
    
    # Create image with minimum width
    image = torch.rand(2, 2, 836, 28)
    
    # Process through encoder
    output = encoder(image)
    
    # Should produce exactly 1 frame
    assert output.shape[0] == 2  # batch size
    assert output.shape[1] == 1  # single frame
    assert output.shape[2] == 64  # 64 values per token
    
    # Values should be finite (raw logits)
    assert torch.all(torch.isfinite(output))


def test_stride_1():
    """Test with stride=1."""
    encoder = WhipstrEncoder(stride=1)
    
    # Create spectrogram with width 56
    image = torch.rand(1, 2, 836, 56)
    
    # Process through encoder
    output = encoder(image)
    
    # Expected frames: (56 - 28) // 1 + 1 = 29
    expected_frames = 29
    assert output.shape[1] == expected_frames, \
        f"Expected {expected_frames} frames, got {output.shape[1]}"


def test_stride_5():
    """Test with stride=5."""
    encoder = WhipstrEncoder(stride=5)
    
    # Create spectrogram with width 100
    image = torch.rand(1, 2, 836, 100)
    
    # Process through encoder
    output = encoder(image)
    
    # Expected frames: (100 - 28) // 5 + 1 = 15
    expected_frames = 15
    assert output.shape[1] == expected_frames, \
        f"Expected {expected_frames} frames, got {output.shape[1]}"


def test_output_shape_matches_expectations():
    """Test output shape matches expectations for various inputs."""
    test_cases = [
        (28, 1, 1),    # min width, stride 1 -> 1 frame
        (56, 1, 29),   # 2 segments, stride 1 -> 29 frames
        (140, 5, 23),  # 5 segments, stride 5 -> 23 frames
        (280, 10, 26), # 10 segments, stride 10 -> 26 frames
    ]
    
    for width, stride, expected_frames in test_cases:
        encoder = WhipstrEncoder(stride=stride)
        image = torch.rand(1, 2, 836, width)
        output = encoder(image)
        
        assert output.shape == (1, expected_frames, 64), \
            f"For width={width}, stride={stride}: " \
            f"expected shape (1, {expected_frames}, 64), got {output.shape}"


def test_invalid_input_channels():
    """Test that invalid number of channels raises ValueError."""
    encoder = WhipstrEncoder(stride=1)
    
    # Create image with wrong number of channels
    image = torch.rand(1, 3, 836, 56)  # 3 channels instead of 2
    
    with pytest.raises(ValueError, match="Expected 2 channels"):
        encoder(image)


def test_invalid_input_height():
    """Test that invalid height raises ValueError."""
    encoder = WhipstrEncoder(stride=1)
    
    # Create image with wrong height
    image = torch.rand(1, 2, 56, 56)  # height 56 instead of 836
    
    with pytest.raises(ValueError, match="Expected height 836"):
        encoder(image)


def test_invalid_input_width():
    """Test that width < 28 raises ValueError."""
    encoder = WhipstrEncoder(stride=1)
    
    # Create image with width < 28
    image = torch.rand(1, 2, 836, 20)
    
    with pytest.raises(ValueError, match="Width must be >= window_size"):
        encoder(image)


def test_invalid_stride_negative():
    """Test that stride < 1 raises ValueError."""
    with pytest.raises(ValueError, match="stride must be >= 1"):
        WhipstrEncoder(stride=0)


def test_invalid_stride_too_large():
    """Test that stride > 28 raises ValueError."""
    with pytest.raises(ValueError, match="stride must be <= window_size"):
        WhipstrEncoder(stride=29)


def test_batch_greater_than_one_with_multiple_chunks():
    """
    Regression test: batch_size > 1 with T > chunk must not scramble windows.
    
    The encoder processes windows in chunks; a bug in the reshape/concat logic
    silently permutes windows between batch elements when B > 1 and T > chunk.
    This test compares single-chunk (always correct) vs multi-chunk output.
    """
    batch_size = 3
    width = 200                                  # gives T = (200-28)//1+1 = 173 > 16
    encoder = WhipstrEncoder(stride=1)
    encoder.eval()

    image = torch.rand(batch_size, 2, 836, width)

    with torch.no_grad():
        y_one_chunk = encoder(image, chunk=10_000)   # single chunk
        y_many_chunks = encoder(image, chunk=16)     # multiple chunks

    torch.testing.assert_close(y_one_chunk, y_many_chunks)
