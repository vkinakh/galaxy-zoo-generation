from typing import Dict
from abc import ABC, abstractmethod

from src.utils import get_device, get_config
from src.utils import SummaryWriterWithSources


class BaseEvaluator(ABC):

    def __init__(self, config_path: str):
        self._config = get_config(config_path)
        self._device = get_device()
        self._writer = SummaryWriterWithSources(
            files_to_copy=[config_path],
            experiment_name=self._config['comment']
        )

    @abstractmethod
    def evaluate(self) -> None:
        pass

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def device(self):
        return self._device

    @abstractmethod
    def _load_model(self):
        pass

    def _log(self, tag: str, message: Dict[str, float], step: int) -> None:
        for key, val in message.items():
            self._writer.add_scalar(f'{tag}/{key}', val, step)
