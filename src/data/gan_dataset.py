import torch
from torch.utils.data import Dataset


class GANDataset(Dataset):
    """Dataset that returns generated images"""

    def __init__(self, model,
                 n: int):
        self._model = model
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i: int) -> torch.Tensor:
        return self._model.sample(1).squeeze()
