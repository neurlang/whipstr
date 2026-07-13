import torch
import pytest
from whipstr.hf_integration import WhipstrConfig, WhipstrForConditionalGeneration


@pytest.fixture
def model():
    config = WhipstrConfig(
        vocab_size=11, stride=1, window_size=11,
        d_model=64, nhead=4, num_encoder_layers=2,
        num_decoder_layers=2, dim_feedforward=256, dropout=0.1,
        encoder_embed_dim=64,
    )
    m = WhipstrForConditionalGeneration(config)
    m.eval()
    return m


def make_input_features(batch_size, width):
    return torch.rand(batch_size, 2, 836, width)


class TestConvertAttentionMask:
    def test_none(self, model):
        assert model._convert_attention_mask(None) is None

    def test_all_valid(self, model):
        batch_size = 2
        W = 50
        attention_mask = torch.ones(batch_size, W, dtype=torch.long)
        result = model._convert_attention_mask(attention_mask)
        expected_T = (W - model.config.window_size) // model.config.stride + 1
        assert result.shape == (batch_size, expected_T)
        assert result.sum().item() == 0  # all False (no padding)

    def test_all_padding(self, model):
        batch_size = 2
        W = 50
        attention_mask = torch.zeros(batch_size, W, dtype=torch.long)
        result = model._convert_attention_mask(attention_mask)
        expected_T = (W - model.config.window_size) // model.config.stride + 1
        assert result.shape == (batch_size, expected_T)
        assert result.sum().item() == batch_size * expected_T  # all True

    def test_partial_padding(self, model):
        batch_size = 2
        W = 50
        attention_mask = torch.ones(batch_size, W, dtype=torch.long)
        valid_frames = 30
        attention_mask[:, valid_frames:] = 0

        result = model._convert_attention_mask(attention_mask)
        expected_T = (W - model.config.window_size) // model.config.stride + 1

        # Tokens whose window starts at or beyond valid_frames are padding
        # Window i covers [i*stride, i*stride + window_size) = [i, i+11)
        # Token i is padding if start i >= valid_frames (no valid frames left)
        # Actually, it's padding if all positions in window are 0
        # Window [valid_frames - window_size + 1, valid_frames) has partial valid frames,
        # so it depends on whether any valid frame is in the window
        # The last valid token has window starting at valid_frames - window_size = 19
        # So tokens 0..19 have at least one valid frame, tokens 20+ are all padding
        assert result.shape == (batch_size, expected_T)
        first_padding = (result[0] == True).nonzero(as_tuple=True)[0]
        if len(first_padding) > 0:
            first_pad_idx = first_padding[0].item()
            assert first_pad_idx >= valid_frames - model.config.window_size + 1


class TestForwardWithAttentionMask:
    def test_no_mask(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        labels = torch.randint(0, 11, (B, 5))
        with torch.no_grad():
            out = model.forward(input_features, labels=labels)
        assert "logits" in out

    def test_with_mask(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        labels = torch.randint(0, 11, (B, 5))
        attention_mask = torch.ones(B, W, dtype=torch.long)
        with torch.no_grad():
            out = model.forward(input_features, labels=labels, attention_mask=attention_mask)
        assert "logits" in out

    def test_no_labels(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        attention_mask = torch.ones(B, W, dtype=torch.long)
        with torch.no_grad():
            out = model.forward(input_features, labels=None, attention_mask=attention_mask)
        assert "logits" in out


class TestGenerateWithAttentionMask:
    def test_no_mask(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        with torch.no_grad():
            out = model.generate(input_features, max_length=5)
        assert out.shape == (B, 5)

    def test_with_mask(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        attention_mask = torch.ones(B, W, dtype=torch.long)
        with torch.no_grad():
            out = model.generate(input_features, max_length=5, attention_mask=attention_mask)
        assert out.shape == (B, 5)
        assert torch.all(out >= 0) and torch.all(out < 11)

    def test_mask_kwargs_passthrough(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        attention_mask = torch.ones(B, W, dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                input_features, max_length=5,
                attention_mask=attention_mask, start_token=0
            )
        assert out.shape == (B, 5)
        assert torch.all(out >= 0) and torch.all(out < 11)
