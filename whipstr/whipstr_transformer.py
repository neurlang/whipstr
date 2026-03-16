import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length to support
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WhipstrTransformer(nn.Module):
    """Transformer encoder-decoder for converting CNN tokens into transcription predictions.
    
    Uses attention mechanisms to convert noisy, over-complete token sequences
    from the CNN encoder into clean predictions through auto-regressive generation.
    """
    
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, vocab_size=11, input_values=64):
        """Initialize the WhipstrTransformer.
        
        Args:
            d_model: Dimension of transformer embeddings (default: 256)
            nhead: Number of attention heads (default: 8)
            num_encoder_layers: Number of transformer encoder layers (default: 4)
            num_decoder_layers: Number of transformer decoder layers (default: 4)
            dim_feedforward: Dimension of feedforward network (default: 1024)
            dropout: Dropout rate (default: 0.1)
            vocab_size: Size of vocabulary including special tokens (default: 11 for 0-9 classes + start token)
        """
        super(WhipstrTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.input_values = input_values
        
        # Encoder path: project tokens from input_values dimensions to d_model
        self.encoder_projection = nn.Linear(input_values, d_model)
        self.encoder_pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Decoder path: embed tokens with configurable vocabulary size
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection to vocabulary logits
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, encoder_tokens, target_digits, target_mask=None):
        """Forward pass with teacher forcing.
        
        Args:
            encoder_tokens: torch.Tensor [batch, T, self.input_values] from CNN encoder
            target_digits: torch.LongTensor [batch, N] previously generated tokens
            target_mask: torch.Tensor [N, N] causal mask for auto-regressive generation
        
        Returns:
            torch.Tensor [batch, N, vocab_size] logits for each position
        """
        # Validate encoder tokens
        if not isinstance(encoder_tokens, torch.Tensor):
            raise TypeError(f"encoder_tokens must be a torch.Tensor, got {type(encoder_tokens).__name__}")
        
        if encoder_tokens.dim() != 3:
            raise ValueError(f"encoder_tokens must be 3D [batch, T, self.input_values], got {encoder_tokens.dim()}D tensor with shape {encoder_tokens.shape}")
        if encoder_tokens.size(2) != self.input_values:
            raise ValueError(f"encoder_tokens must have self.input_values features (class scores), got {encoder_tokens.size(2)}")
        
        # Note: encoder_tokens are raw logits (no range restriction)
        
        # Check for NaN or Inf
        if torch.isnan(encoder_tokens).any():
            raise ValueError("encoder_tokens contains NaN values")
        if torch.isinf(encoder_tokens).any():
            raise ValueError("encoder_tokens contains Inf values")
        
        # Validate target sequence
        if not isinstance(target_digits, torch.Tensor):
            raise TypeError(f"target_digits must be a torch.Tensor, got {type(target_digits).__name__}")
        
        if target_digits.dim() != 2:
            raise ValueError(f"target_digits must be 2D [batch, N], got {target_digits.dim()}D tensor with shape {target_digits.shape}")
        if torch.any(target_digits < 0) or torch.any(target_digits >= self.vocab_size):
            min_val = target_digits.min().item()
            max_val = target_digits.max().item()
            raise ValueError(f"target_digits must have values in range [0, {self.vocab_size-1}], got range [{min_val}, {max_val}]")
        
        # Check device mismatch
        if encoder_tokens.device != target_digits.device:
            raise RuntimeError(
                f"Device mismatch: encoder_tokens is on {encoder_tokens.device} "
                f"but target_digits is on {target_digits.device}. "
                f"All tensors must be on the same device."
            )
        
        batch_size = encoder_tokens.size(0)
        
        # Validate batch sizes match
        if target_digits.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: encoder_tokens has batch size {batch_size} "
                f"but target_digits has batch size {target_digits.size(0)}"
            )
        
        try:
            # Encoder path
            encoder_out = self.encoder_projection(encoder_tokens)  # [batch, T, d_model]
            encoder_out = self.encoder_pos_encoding(encoder_out)
            encoder_memory = self.transformer_encoder(encoder_out)  # [batch, T, d_model]
            
            # Decoder path
            decoder_input = self.decoder_embedding(target_digits)  # [batch, N, d_model]
            decoder_input = self.decoder_pos_encoding(decoder_input)
            
            # Create causal mask if not provided
            if target_mask is None:
                target_len = target_digits.size(1)
                target_mask = self._generate_square_subsequent_mask(target_len).to(encoder_tokens.device)
            
            # Decode
            decoder_out = self.transformer_decoder(
                decoder_input,
                encoder_memory,
                tgt_mask=target_mask
            )  # [batch, N, d_model]
            
            # Project to vocabulary logits
            logits = self.output_projection(decoder_out)  # [batch, N, vocab_size]
            
            return logits
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"CUDA out of memory during transformer forward pass. "
                    f"Input shapes: encoder_tokens={encoder_tokens.shape}, target_digits={target_digits.shape}. "
                    f"Try reducing batch size or sequence length. Original error: {str(e)}"
                ) from e
            else:
                raise RuntimeError(f"Error during transformer forward pass: {str(e)}") from e
    
    def generate(self, encoder_tokens, max_length, start_token):
        """Auto-regressive generation for inference.
        
        Args:
            encoder_tokens: torch.Tensor [batch, T, self.input_values] from CNN encoder
            max_length: Maximum number of tokens to generate
            start_token: Special token to begin generation (default: self.input_values for backward compatibility)
        
        Returns:
            torch.LongTensor [batch, max_length] predicted tokens
        """
        batch_size = encoder_tokens.size(0)
        device = encoder_tokens.device
        
        # Validate encoder tokens
        if encoder_tokens.dim() != 3:
            raise ValueError(f"encoder_tokens must be 3D [batch, T, self.input_values], got shape {encoder_tokens.shape}")
        if encoder_tokens.size(2) != self.input_values:
            raise ValueError(f"encoder_tokens must have self.input_values features, got {encoder_tokens.size(2)}")
        
        # Encode once
        encoder_out = self.encoder_projection(encoder_tokens)
        encoder_out = self.encoder_pos_encoding(encoder_out)
        encoder_memory = self.transformer_encoder(encoder_out)
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Generate tokens one at a time
        for _ in range(max_length):
            # Embed current sequence
            decoder_input = self.decoder_embedding(generated)
            decoder_input = self.decoder_pos_encoding(decoder_input)
            
            # Create causal mask
            tgt_len = generated.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)
            
            # Decode
            decoder_out = self.transformer_decoder(
                decoder_input,
                encoder_memory,
                tgt_mask=tgt_mask
            )
            
            # Get logits for last position
            logits = self.output_projection(decoder_out[:, -1, :])  # [batch, vocab_size]
            
            # Sample next token (greedy)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        # Remove start token and return only generated output
        return generated[:, 1:]  # [batch, max_length]
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for auto-regressive generation.
        
        Args:
            sz: Size of the square mask
        
        Returns:
            torch.Tensor [sz, sz] with -inf in upper triangle
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
