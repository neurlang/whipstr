"""SpectrogramWindowDataset — placeholder, implemented in task 5."""

from torch.utils.data import Dataset


class SpectrogramWindowDataset(Dataset):
    """Slices full spectrograms into (2, 11, 836) windows for Stage 2 training."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SpectrogramWindowDataset is implemented in task 5")
