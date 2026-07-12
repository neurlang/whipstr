import torch
from torch.utils.data import Dataset
import os
from phase import Phase
from tqdm import tqdm


class WhipstrTSVSpeechDataset(Dataset):
    """
    Dataset for Whipstr ASR/STT using TSV speech data.
    
    Loads FLAC audio files and their transcriptions from a TSV file,
    converts audio to phase spectrograms, and formats them for Whipstr training.
    """
    
    def __init__(self, tsv_path, limit=-1, all_chars=None):
        """
        Initialize the Whipstr TSV Speech dataset.
        
        Args:
            tsv_path: Path to TSV file with format: <flac_path>\t<transcription>
            limit: Limit file count (default: 0)
        """
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")
        
        if not isinstance(limit, (int, float)):
            raise TypeError(f"limit must be a number, got {type(limit).__name__}")
        if limit < -1:
            raise ValueError(f"limit must be >= -1, got {limit}")
        
        self.tsv_path = tsv_path
        self.limit = limit
        self.phase = Phase(y_reverse=True)
        
        # Load TSV data
        self.samples = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset..."):
                line = line.strip()
                if not line:
                    continue
                parts = []
                if '\t' in line:
                    parts = line.split('\t')
                elif ',' in line:
                    parts = line.split(',')
                elif '|' in line:
                    parts = line.split('|')
                else:
                    parts = [line]
                if len(parts) == 0:
                    continue

                # If flac_path is a glob pattern like *.wav, expand to directory files

                if parts[0].endswith(('*.wav', '*.flac', '*.mp3')):
                    self._add_directory_files(parts[0])
                    continue

                if len(parts) != 2:
                    raise ValueError(f"Invalid TSV line format (expected 2 columns): {line}")
                flac_path, transcription = parts
                
                if not os.path.exists(flac_path):
                    raise FileNotFoundError(f"FLAC file not found: {flac_path}")

                if all_chars is not None:
                    all_chars.update(transcription)
                
                if self.limit != 0:
                    self.limit -= 1        
                    self.samples.append((flac_path, transcription))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in TSV file: {tsv_path}")

    def _add_directory_files(self, pattern_path):
        """Add all files from a directory matching the glob suffix."""
        directory = os.path.dirname(pattern_path)
        suffix = os.path.basename(pattern_path).lstrip('*')  # e.g. '.wav'
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        for fname in os.listdir(directory):
            if fname.endswith(suffix):
                full_path = os.path.join(directory, fname)
                if self.limit != 0:
                    self.limit -= 1
                    self.samples.append((full_path, ''))
                else:
                    return
    

    def chars_for_subset(self, indices):
        """Return the set of characters appearing in transcriptions for given indices.

        Operates on stored transcriptions only — no audio loading.
        """
        chars = set()
        for idx in indices:
            chars.update(self.samples[idx][1])
        return chars

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image_tensor, transcription)
                - image_tensor: torch.Tensor of shape [2, 836, W] with values in [0, 1]
                - transcription: str containing the ground truth text
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
        
        flac_path, transcription = self.samples[idx]
        
        try:
            # Convert FLAC to tensor using phase spectrogram
            audio_array = self.phase.to_tensor_flac(flac_path)
                        
            # Convert numpy array to PyTorch tensor
            if not isinstance(audio_array, torch.Tensor):
                audio_tensor = torch.from_numpy(audio_array).float()
            else:
                audio_tensor = audio_array.float()
            
            # Reshape from (time_frames, 2) to (2, num_freqs, actual_frames)
            # The data is encoded as flattened: total_samples = num_freqs * actual_frames
            num_freqs = self.phase.num_freqs
            if num_freqs == 0:
                raise ValueError("num_freqs is 0 - ensure to_tensor_flac was called first")
            
            total_samples = audio_tensor.shape[0]
            actual_frames = total_samples // num_freqs
            
            # Reshape: (total_samples, 2) -> (actual_frames, num_freqs, 2) -> (2, num_freqs, actual_frames)
            audio_tensor = audio_tensor.reshape(actual_frames, num_freqs, 2)
            audio_tensor = audio_tensor.permute(2, 1, 0)  # (2, num_freqs, actual_frames)
            
            # Get dimensions
            channels, height, width = audio_tensor.shape
            
            # Pad height to 836 pixels if needed (pad at bottom with zeros)
            if height != 836:
                if height < 836:
                    # Pad at the bottom with zeros
                    pad_amount = 836 - height
                    padding = torch.zeros(channels, pad_amount, width)
                    audio_tensor = torch.cat([audio_tensor, padding], dim=1)  # (2, 836, width)
                else:
                    # If somehow larger than 836, crop from bottom
                    audio_tensor = audio_tensor[:, :836, :]
                        
            # Return raw float values without normalization
            red_channel = audio_tensor[0]
            green_channel = audio_tensor[1]
            
            # Stack to create [2, 836, W] tensor
            image_tensor = torch.stack([red_channel, green_channel], dim=0)
            
            return image_tensor, transcription
            
        except Exception as e:
            raise RuntimeError(f"Error loading sample at index {idx} (file: {flac_path}): {str(e)}") from e
