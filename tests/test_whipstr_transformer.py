import torch
from hypothesis import given, strategies as st, settings
from whipstr.whipstr_transformer import WhipstrTransformer
import pytest

# Default vocab_size for the transformer is 11 (0-9 classes + start token)
DEFAULT_VOCAB_SIZE = 11


# Property 10: Token input validation
# Feature: whipstr-stt-asr, Property 10: Token input validation
# Validates: Requirements 4.1
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=100),
    target_len=st.integers(min_value=1, max_value=20)
)
def test_property_10_token_input_validation(batch_size, seq_len, target_len):
    """
    Property 10: Token input validation
    For any token sequence input to the transformer, each token should have
    exactly 10 elements (raw logits, no range restriction).
    """
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    # Create valid encoder tokens (raw logits can be any finite value)
    encoder_tokens = torch.randn(batch_size, seq_len, 64)

    # Create valid target sequence
    target_digits = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, target_len))

    # Should process without errors
    try:
        output = transformer(encoder_tokens, target_digits)

        assert isinstance(output, torch.Tensor), \
            f"Output should be a tensor, got {type(output)}"
        assert output.shape[0] == batch_size, \
            f"Expected batch size {batch_size}, got {output.shape[0]}"
        assert output.shape[1] == target_len, \
            f"Expected target length {target_len}, got {output.shape[1]}"
        assert output.shape[2] == DEFAULT_VOCAB_SIZE, \
            f"Expected {DEFAULT_VOCAB_SIZE} output values, got {output.shape[2]}"

    except Exception as e:
        pytest.fail(f"Transformer failed with valid inputs: {e}")

    # Test that wrong number of features raises ValueError
    invalid_tokens_wrong_size = torch.rand(batch_size, seq_len, 5)  # 5 instead of 64
    with pytest.raises(ValueError, match="encoder_tokens must have self.input_values features"):
        transformer(invalid_tokens_wrong_size, target_digits)


# Property 11: Auto-regressive interface
# Feature: whipstr-stt-asr, Property 11: Auto-regressive interface
# Validates: Requirements 4.3
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=10, max_value=50),
    max_length=st.integers(min_value=1, max_value=15)
)
def test_property_11_autoregressive_interface(batch_size, seq_len, max_length):
    """
    Property 11: Auto-regressive interface
    For any generation step, the transformer should accept previously generated
    character indices as input and produce the next prediction.
    """
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    transformer.eval()

    encoder_tokens = torch.rand(batch_size, seq_len, 64)

    with torch.no_grad():
        generated = transformer.generate(encoder_tokens, max_length=max_length)

    assert generated.shape == (batch_size, max_length), \
        f"Expected shape ({batch_size}, {max_length}), got {generated.shape}"

    # Generated values should be in range [0, vocab_size-1]
    assert torch.all(generated >= 0), \
        f"Found generated values < 0: min = {generated.min().item()}"
    assert torch.all(generated < DEFAULT_VOCAB_SIZE), \
        f"Found generated values >= {DEFAULT_VOCAB_SIZE}: max = {generated.max().item()}"

    assert generated.dtype == torch.long, \
        f"Expected dtype torch.long, got {generated.dtype}"


# Property 12: Output format
# Feature: whipstr-stt-asr, Property 12: Output format
# Validates: Requirements 4.4
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=100),
    target_len=st.integers(min_value=1, max_value=20)
)
def test_property_12_output_format(batch_size, seq_len, target_len):
    """
    Property 12: Output format
    For any input sequence, the transformer output should have shape [batch, N, vocab_size].
    """
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    encoder_tokens = torch.rand(batch_size, seq_len, 64)
    target_digits = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, target_len))

    output = transformer(encoder_tokens, target_digits)

    expected_shape = (batch_size, target_len, DEFAULT_VOCAB_SIZE)
    assert output.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {output.shape}"

    assert output.dtype in [torch.float32, torch.float64], \
        f"Expected float dtype, got {output.dtype}"


# Property 13: Prediction range
# Feature: whipstr-stt-asr, Property 13: Prediction range
# Validates: Requirements 4.5
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=100),
    target_len=st.integers(min_value=1, max_value=20)
)
def test_property_13_prediction_range(batch_size, seq_len, target_len):
    """
    Property 13: Prediction range
    For any transformer output, applying argmax should produce values
    in the range [0, vocab_size-1].
    """
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    encoder_tokens = torch.rand(batch_size, seq_len, 64)
    target_digits = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, target_len))

    output = transformer(encoder_tokens, target_digits)
    predictions = torch.argmax(output, dim=-1)

    assert torch.all(predictions >= 0), \
        f"Found predictions < 0: min = {predictions.min().item()}"
    assert torch.all(predictions < DEFAULT_VOCAB_SIZE), \
        f"Found predictions >= {DEFAULT_VOCAB_SIZE}: max = {predictions.max().item()}"

    assert predictions.shape == (batch_size, target_len), \
        f"Expected shape ({batch_size}, {target_len}), got {predictions.shape}"


# Unit Tests for edge cases

def test_short_sequence_n3():
    """Test with short sequences (N=3)."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    batch_size = 2
    seq_len = 10
    target_len = 3

    encoder_tokens = torch.rand(batch_size, seq_len, 64)
    target_digits = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, target_len))

    output = transformer(encoder_tokens, target_digits)

    assert output.shape == (batch_size, target_len, DEFAULT_VOCAB_SIZE), \
        f"Expected shape ({batch_size}, {target_len}, {DEFAULT_VOCAB_SIZE}), got {output.shape}"

    predictions = torch.argmax(output, dim=-1)
    assert torch.all(predictions >= 0) and torch.all(predictions < DEFAULT_VOCAB_SIZE)


def test_longer_sequence_n15():
    """Test with longer sequences (N=15)."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    batch_size = 2
    seq_len = 50
    target_len = 15

    encoder_tokens = torch.rand(batch_size, seq_len, 64)
    target_digits = torch.randint(0, DEFAULT_VOCAB_SIZE, (batch_size, target_len))

    output = transformer(encoder_tokens, target_digits)

    assert output.shape == (batch_size, target_len, DEFAULT_VOCAB_SIZE), \
        f"Expected shape ({batch_size}, {target_len}, {DEFAULT_VOCAB_SIZE}), got {output.shape}"

    predictions = torch.argmax(output, dim=-1)
    assert torch.all(predictions >= 0) and torch.all(predictions < DEFAULT_VOCAB_SIZE)


def test_generate_produces_valid_digits():
    """Test that generate method produces valid output characters."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    transformer.eval()

    batch_size = 3
    seq_len = 30
    max_length = 10

    encoder_tokens = torch.rand(batch_size, seq_len, 64)

    with torch.no_grad():
        generated = transformer.generate(encoder_tokens, max_length=max_length)

    assert generated.shape == (batch_size, max_length), \
        f"Expected shape ({batch_size}, {max_length}), got {generated.shape}"

    assert torch.all(generated >= 0) and torch.all(generated < DEFAULT_VOCAB_SIZE), \
        f"Generated invalid values: min={generated.min().item()}, max={generated.max().item()}"

    assert generated.dtype == torch.long


def test_invalid_target_digits():
    """Test that invalid target digits raise ValueError."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    encoder_tokens = torch.rand(2, 10, 64)

    # Target values >= vocab_size
    invalid_targets = torch.randint(11, 20, (2, 5))

    with pytest.raises(ValueError, match="target_digits must have values in range"):
        transformer(encoder_tokens, invalid_targets)


def test_invalid_encoder_tokens_shape():
    """Test that invalid encoder token shape raises ValueError."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    invalid_tokens = torch.rand(10, 10)
    target_digits = torch.randint(0, DEFAULT_VOCAB_SIZE, (2, 5))

    with pytest.raises(ValueError, match="encoder_tokens must be 3D"):
        transformer(invalid_tokens, target_digits)


def test_invalid_target_digits_shape():
    """Test that invalid target digits shape raises ValueError."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    encoder_tokens = torch.rand(2, 10, 64)

    invalid_targets = torch.randint(0, DEFAULT_VOCAB_SIZE, (5,))

    with pytest.raises(ValueError, match="target_digits must be 2D"):
        transformer(encoder_tokens, invalid_targets)


def test_generate_with_different_start_token():
    """Test generate with different start token."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    transformer.eval()

    encoder_tokens = torch.rand(2, 20, 64)

    with torch.no_grad():
        generated1 = transformer.generate(encoder_tokens, max_length=5, start_token=10)

    assert generated1.shape == (2, 5)
    assert torch.all(generated1 >= 0) and torch.all(generated1 < DEFAULT_VOCAB_SIZE)


def test_causal_mask_generation():
    """Test that causal mask is properly generated."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)

    mask = transformer._generate_square_subsequent_mask(5)

    assert mask.shape == (5, 5)

    for i in range(5):
        for j in range(5):
            if j > i:
                assert mask[i, j] == float('-inf'), \
                    f"Expected -inf at position ({i}, {j}), got {mask[i, j]}"
            else:
                assert mask[i, j] == 0.0, \
                    f"Expected 0.0 at position ({i}, {j}), got {mask[i, j]}"


def test_generate_with_padding_mask():
    """Test that generate() accepts and uses encoder_padding_mask."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    transformer.eval()

    batch_size = 3
    seq_len = 20
    max_length = 8

    encoder_tokens = torch.rand(batch_size, seq_len, 64)
    # Mask last 5 positions for all samples
    encoder_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    encoder_padding_mask[:, -5:] = True

    with torch.no_grad():
        generated = transformer.generate(
            encoder_tokens, max_length=max_length,
            encoder_padding_mask=encoder_padding_mask
        )

    assert generated.shape == (batch_size, max_length), \
        f"Expected shape ({batch_size}, {max_length}), got {generated.shape}"
    assert torch.all(generated >= 0) and torch.all(generated < DEFAULT_VOCAB_SIZE), \
        f"Generated invalid values: min={generated.min().item()}, max={generated.max().item()}"
    assert generated.dtype == torch.long


def test_generate_padding_mask_all_valid():
    """Test that generate() with all-valid mask produces same output as without mask."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    transformer.eval()

    batch_size = 2
    seq_len = 15
    max_length = 6

    encoder_tokens = torch.rand(batch_size, seq_len, 64)
    all_valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        out_no_mask = transformer.generate(encoder_tokens, max_length=max_length)
        out_with_mask = transformer.generate(
            encoder_tokens, max_length=max_length,
            encoder_padding_mask=all_valid_mask
        )

    assert torch.equal(out_no_mask, out_with_mask), \
        "All-valid mask should produce identical output to no mask"


def test_generate_padding_mask_no_crash():
    """Test that generate() doesn't crash with edge-case masks."""
    transformer = WhipstrTransformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    transformer.eval()

    batch_size = 2
    seq_len = 10
    max_length = 5

    encoder_tokens = torch.rand(batch_size, seq_len, 64)

    # Mask first half
    first_half = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    first_half[:, :seq_len // 2] = True
    with torch.no_grad():
        out = transformer.generate(
            encoder_tokens, max_length=max_length,
            encoder_padding_mask=first_half
        )
    assert out.shape == (batch_size, max_length)
    assert torch.all(out >= 0) and torch.all(out < DEFAULT_VOCAB_SIZE)

    # Single element mask for each item
    single_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    single_mask[:, 0] = True
    with torch.no_grad():
        out = transformer.generate(
            encoder_tokens, max_length=max_length,
            encoder_padding_mask=single_mask
        )
    assert out.shape == (batch_size, max_length)
    assert torch.all(out >= 0) and torch.all(out < DEFAULT_VOCAB_SIZE)
