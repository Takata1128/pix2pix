import importlib
import torch.utils.data
from dataset.base_dataset import BaseDataset
from config import Config


def find_dataset_using_name(dataset_name) -> BaseDataset:
    dataset_filename = "dataset." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"

    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name)
        )

    return dataset


def create_dataset(config: Config) -> BaseDataset:
    data_loader = CustomDatasetDataLoader(config)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    def __init__(self, config: Config):
        self.config = config
        dataset_class = find_dataset_using_name(config.dataset_mode)
        self.dataset = dataset_class(config)
        print("dataset {} was created".format(type(self.dataset).__name__))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=not config.serial_batches,
            num_workers=int(config.num_threads),
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.config.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.config.batch_size >= self.config.max_dataset_size:
                break
            yield data
