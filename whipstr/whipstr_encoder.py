import torch
import torch.nn as nn


class WhipstrEncoder(nn.Module):
    """CNN encoder that processes overlapping spectrogram windows for Whipstr STT (ASR).
    
    Extracts acoustic features from overlapping spectrogram windows using a shared CNN,
    producing a sequence of tokens where each token contains 64 values
    representing output class scores.
    """
    
    def __init__(self, stride=1, window_size=28, window_height=836, output_values=64):
        """Initialize the WhipstrEncoder.
        
        Args:
            stride: Pixel shift between consecutive windows (default: 1)
            window_size: Width of each window (default: 28)
            window_height: Height of each window (default: 836 for stretched spectrograms)
        """
        super(WhipstrEncoder, self).__init__()
        
        # Validate inputs
        if not isinstance(stride, int):
            raise TypeError(f"stride must be an integer, got {type(stride).__name__}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if stride > window_size:
            raise ValueError(f"stride must be <= window_size, got stride={stride}, window_size={window_size}")
        
        if not isinstance(window_size, int):
            raise TypeError(f"window_size must be an integer, got {type(window_size).__name__}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        
        if not isinstance(window_height, int):
            raise TypeError(f"window_height must be an integer, got {type(window_height).__name__}")
        if window_height != 836:
            raise ValueError(f"window_height must be 836 for stretched spectrograms, got {window_height}")
        
        self.stride = stride
        self.window_size = window_size
        self.window_height = window_height
        self.output_values = output_values
        
        # CNN architecture for processing [836, window_size] spectrogram windows
        # Input: [batch, 2, 836, window_size]
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # -> [batch, 64, 420, window_size//2]
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # -> [batch, 128, 210, window_size//4]
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)  # -> [batch, 256, 105, window_size//8]
        
        # Calculate flattened feature size after pooling
        # Height: 832 -> 416 -> 208 -> 104
        # Width: window_size -> window_size//2 -> window_size//4 -> window_size//8
        final_height = 832
        final_width = window_size // 8
        fc_input_size = 32 * final_height * final_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, output_values)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, image, chunk=16):
        """Process spectrogram through overlapping windows.
        
        Args:
            image: torch.Tensor of shape [batch, 2, 836, W]
        
        Returns:
            torch.Tensor of shape [batch, T, output_values] where T = (W - window_size) // stride + 1
        """
        # Validate input is a tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(image).__name__}")
        
        # Validate input dimensions
        if image.dim() != 4:
            raise ValueError(f"Input must be 4D [batch, channels, height, width], got {image.dim()}D tensor with shape {image.shape}")
        
        batch_size, channels, height, width = image.shape
        
        # Validate input shape
        if channels != 2:
            raise ValueError(f"Expected 2 channels, got {channels}. "
                           f"Input shape should be [batch, 2, 836, W]")
        if height != 836:
            raise ValueError(f"Expected height 836 (spectrogram height), got {height}. "
                           f"Input shape should be [batch, 2, 836, W]")
        if width < self.window_size:
            raise ValueError(f"Width must be >= window_size ({self.window_size}), got {width}. "
                           f"Cannot extract {self.window_size}-wide windows from spectrogram with width {width}")
        
        # Check for NaN or Inf values
        if torch.isnan(image).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(image).any():
            raise ValueError("Input contains Inf values")
        
        try:

            outputs = []

            windows = image.unfold(3, self.window_size, self.stride)
            windows = windows.permute(0, 3, 1, 2, 4)  # DO NOT make contiguous yet

            num_windows = windows.shape[1]

            for start in range(0, num_windows, chunk):

                end = min(start + chunk, num_windows)

                w = windows[:, start:end]                       # [B, local_t, 2, 836, ws]
                B_local, local_t = w.shape[0], w.shape[1]
                w = w.contiguous().view(B_local * local_t, 2, 836, self.window_size)

                x = self.relu(self.conv1(w))
                x = self.relu(self.conv2(x))
                x = self.pool1(x)

                x = self.relu(self.conv3(x))
                x = self.pool2(x)

                x = self.relu(self.conv4(x))
                x = self.pool3(x)

                x = x.view(x.size(0), -1)

                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)

                outputs.append(x.view(B_local, local_t, self.output_values))

            x = torch.cat(outputs, dim=1)

            return x
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"CUDA out of memory while processing image of shape {image.shape}. "
                    f"Try reducing batch size or image width. Original error: {str(e)}"
                ) from e
            else:
                raise RuntimeError(f"Error during forward pass: {str(e)}") from e
