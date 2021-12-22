from typing import Union, Optional, Callable, List
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

image_loader = partial(read_image, mode=ImageReadMode.RGB)


class GalaxyZooUnlabeledDataset(Dataset):

    """Unlabeled Galaxy Zoo dataset"""

    def __init__(self,
                 root: Union[str, Path],
                 transform: Optional[Callable] = None,
                 return_mock_label: bool = True):
        """
        Args:
            root: path to folder with images
            transform: transform to apply
            return_mock_label: if True, mock label will be returned
        """

        root = Path(root)
        self._transform = transform
        self._image_paths = [x for x in root.glob('**/*.jpg') if x.is_file()]
        self._return_mock_label = return_mock_label

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, i: int):
        image_path = self._image_paths[i]
        x = image_loader(str(image_path))

        if self._transform is not None:
            x = self._transform(x)

        if self._return_mock_label:
            return x, 0
        return x


class GalaxyZooLabeledDataset(Dataset):

    def __init__(self,
                 root: Union[str, Path],
                 anno_path: Union[str, Path],
                 columns: Optional[List[str]] = None,
                 transform: Optional[Callable] = None):

        root = Path(root)
        self._transform = transform
        self._image_paths = [x for x in root.glob('**/*.jpg') if x.is_file()]
        self._annotations = pd.read_csv(anno_path)

        if columns is not None:
            if 'GalaxyID' not in columns:
                columns.append('GalaxyID')
            self._annotations = self._annotations[columns]
        self._columns = list(self._annotations.columns)[:-1]

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, i: int):
        image_path = self._image_paths[i]
        x = image_loader(str(image_path))

        galaxy_id = int(image_path.stem)
        annotation = self._annotations[self._annotations['GalaxyID'] == galaxy_id]
        annotation = annotation.drop(['GalaxyID'], axis=1)
        annotation = annotation.values[0].astype(np.float32)

        if self._transform is not None:
            x = self._transform(x)
        return x, annotation

    @property
    def columns(self):
        return self._columns
