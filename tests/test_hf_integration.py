import torch
import pytest
from transformers.modeling_outputs import Seq2SeqLMOutput
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

        assert result.shape == (batch_size, expected_T)
        first_padding = (result[0] == True).nonzero(as_tuple=True)[0]
        if len(first_padding) > 0:
            first_pad_idx = first_padding[0].item()
            assert first_pad_idx >= valid_frames - model.config.window_size + 1


class TestForward:
    def test_returns_seq2seq_lm_output(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        with torch.no_grad():
            out = model.forward(input_features)
        assert isinstance(out, Seq2SeqLMOutput)

    def test_no_labels_returns_encoder_only(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        with torch.no_grad():
            out = model.forward(input_features)
        assert out.loss is None
        assert out.logits is None
        assert out.encoder_last_hidden_state is not None

    def test_with_labels_returns_loss_and_logits(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        labels = torch.randint(0, 11, (B, 10))
        with torch.no_grad():
            out = model.forward(input_features, labels=labels)
        assert isinstance(out, Seq2SeqLMOutput)
        assert out.loss is not None
        assert out.logits is not None
        assert out.encoder_last_hidden_state is not None
        assert out.logits.shape == (B, labels.size(1) - 1, 11)

    def test_loss_is_scalar(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        labels = torch.randint(0, 11, (B, 10))
        with torch.no_grad():
            out = model.forward(input_features, labels=labels)
        assert out.loss.ndim == 0

    def test_loss_is_differentiable(self, model):
        model.train()
        B, W = 2, 50
        input_features = make_input_features(B, W).requires_grad_(True)
        labels = torch.randint(0, 11, (B, 10))
        out = model.forward(input_features, labels=labels)
        loss = out.loss
        loss.backward()
        assert input_features.grad is not None

    def test_with_attention_mask(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        labels = torch.randint(0, 11, (B, 10))
        attention_mask = torch.ones(B, W, dtype=torch.long)
        with torch.no_grad():
            out = model.forward(input_features, labels=labels, attention_mask=attention_mask)
        assert isinstance(out, Seq2SeqLMOutput)
        assert out.loss is not None


class TestGenerate:
    def test_no_mask(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        with torch.no_grad():
            out = model.generate(input_features, max_length=5, eos_token=255)
        assert out.shape == (B, 5)

    def test_with_mask(self, model):
        B, W = 2, 50
        input_features = make_input_features(B, W)
        attention_mask = torch.ones(B, W, dtype=torch.long)
        with torch.no_grad():
            out = model.generate(input_features, max_length=5, attention_mask=attention_mask)
        assert out.shape == (B, 5)
        assert torch.all(out >= 0) and torch.all(out < 11)

    def test_kwargs_passthrough(self, model):
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
