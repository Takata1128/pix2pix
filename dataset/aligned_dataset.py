from dataset.base_dataset import BaseDataset
from dataset.utils import make_dataset, get_params, get_transform
from PIL import Image
from config import Config
import os


class AlignedDataset(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self.dir_AB = os.path.join(config.data_root, config.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, config.max_dataset_size))
        assert self.config.load_size >= self.config.crop_size
        self.input_nc = (
            self.config.output_nc
            if self.config.direction == "BtoA"
            else self.config.input_nc
        )
        self.output_nc = (
            self.config.input_nc
            if self.config.direction == "BtoA"
            else self.config.output_nc
        )

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        transform_params = get_params(self.config, A.size)
        A_transform = get_transform(
            self.config, transform_params, grayscale=(self.input_nc == 1)
        )
        B_transform = get_transform(
            self.config, transform_params, grayscale=(self.output_nc == 1)
        )

        A = A_transform(A)
        B = B_transform(B)

        return {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        return len(self.AB_paths)
