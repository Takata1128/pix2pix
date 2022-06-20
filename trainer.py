from config import Config
from dataset import create_dataset
from models import create_model
import wandb
import time
from abc import ABC, abstractmethod


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = create_dataset(config)
        self.dataset_size = len(self.dataset)
        self.model = create_model(config)
        self.model.setup(config)

        self.total_iters = 0

    def train(self):
        with wandb.init(
            project=self.config.project_name, config=self.config.__dict__
        ) as w:
            for epoch in range(
                self.config.epoch_count,
                self.config.n_epochs + self.config.n_epochs_decay + 1,
            ):
                epoch_start_time = time.time()
                self.model.update_learning_rate()

                for i, data in enumerate(self.dataset):
                    self.total_iters += 1

                    self.model.set_input(data)
                    self.model.optimize_parameters()

                    self.log()

                    if self.total_iters % self.config.log_image_freq == 0:
                        self.log_images()

                    if self.total_iters % self.config.save_latest_freq == 0:
                        print(
                            "saving the latest model (epoch {},total_iters {})".format(
                                epoch, self.total_iters
                            )
                        )
                        save_suffix = (
                            "iter_{}".format(self.total_iters)
                            if self.config.save_by_iter
                            else "latest"
                        )

                        self.model.save_networks(save_suffix)
            print(
                "End of epoch %d / %d \t Time Taken: %d sec"
                % (
                    epoch,
                    self.config.n_epochs + self.config.n_epochs_decay,
                    time.time() - epoch_start_time,
                )
            )

    def log(self):
        dict = self.model.get_current_losses()
        wandb.log(dict)

    def log_images(self):
        visual_dict = self.model.get_current_visuals()
        for key, value in visual_dict.items():
            wandb.log({key: wandb.Image(value)})
