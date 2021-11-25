from typing import Optional, Callable, List

from torch.utils.data import Dataset

from src.data import GalaxyZooLabeledDataset
from src.utils import PathOrStr


AVAILABLE_DATASETS = ['galaxy_zoo']


def get_dataset(name: str,
                data_path: PathOrStr,
                anno_file: Optional[PathOrStr] = None,
                transform: Optional[Callable] = None,
                columns: Optional[List[str]] = None) -> Dataset:

    """Returns dataset based on conditions

    Args:
        name: dataset name

        data_path: path to dataset

        anno_file: path to file for annotation

        transform: transform to apply to the data

        columns: list of columns to select (only for CelebA)

    Returns:
        Dataset: loaded dataset
    """

    if name not in AVAILABLE_DATASETS:
        raise ValueError('Unsupported dataset')

    if name == 'galaxy_zoo':
        dataset = GalaxyZooLabeledDataset(data_path, anno_file, columns, transform)
    else:
        raise ValueError('Unsupported dataset')

    return dataset


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch
