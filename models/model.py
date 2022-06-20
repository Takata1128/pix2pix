from typing import OrderedDict
import torch
from torchvision.utils import make_grid
import os
import networks
from abc import ABC, abstractmethod
from config import Config


class BaseModel(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.gpu_ids = config.gpu_ids
        self.is_train = config.is_train
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )
        self.save_dir = os.path.join(config.checkpoints_dir, config.name)
        os.makedirs(os.path.join(config.checkpoints_dir, config.name), exist_ok=True)

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, config):
        if self.is_train:
            self.schedulers = [
                networks.get_scheduler(optimizer, config)
                for optimizer in self.optimizers
            ]
        if not self.is_train or config.continue_train:
            load_suffix = config.epoch
            self.load_networks(load_suffix)

        self.print_networks(config.verbose)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def get_images_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.config.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = {:.7f}".format(lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                t = getattr(self, name)
                visual_ret[name] = make_grid(
                    t[: self.config.visualize_figs], nrow=self.config.visualize_nrow
                )
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, "loss_" + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "{}_net_{}.pth".format(epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "{}_net_{}.pth".format(epoch, name)
                if self.config.is_train and self.config.pretrained_name is not None:
                    load_dir = os.path.join(
                        self.config.checkpoints_dir, self.config.pretrained_name
                    )
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from {}".format(load_path))

                state_dict = torch.load(load_path, map_location=self.device)

                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        print("---------- Networks initialized ----------")
        for name in self.model_names:
            net = getattr(self, "net" + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print(
                "[Network {}] Total number of parameters : {:.3f} M".format(
                    name, num_params / 1e6
                )
            )
        print("------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
