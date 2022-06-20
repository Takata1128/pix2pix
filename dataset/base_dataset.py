import torch.utils.data as data
from abc import ABC, abstractmethod
from config import Config


class BaseDataset(data.Dataset, ABC):
    def __init__(self, config: Config):
        self.config = config
        self.root = config.data_root

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass
