from dataclasses import field, dataclass


@dataclass
class Config:
    project_name: str = "pix2pix"
    data_root: str = "/root/datasets/facades"
    name: str = "pix2pix_facade"
    gpu_ids: list = field(default_factory=lambda: [0])
    checkpoints_dir: str = "./checkpoints"
    model: str = "pix2pix"
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netD: str = "basic"
    netG: str = "unet_256"
    n_layers_D: int = 3
    norm: str = "instance"
    init_type: str = "normal"
    init_gain: float = 0.02
    no_dropout: bool = True
    dataset_mode: str = "aligned"
    direction: str = "AtoB"
    serial_batches: bool = True
    num_threads: int = 4
    batch_size: int = 8
    load_size: int = 286
    crop_size: int = 256
    max_dataset_size: int = float("inf")
    preprocess: list = field(default_factory=lambda: ["resize", "crop"])
    no_flip: bool = True
    visualize_figs = 16
    visualize_nrow = 4
    load_epoch: str = "latest"
    load_iter: int = 0
    verbose: bool = True
    suffix: str = ""

    is_train: bool = True
    save_latest_freq: int = 5000
    log_image_freq: int = 5000
    save_epoch_freq: int = 5

    save_by_iter: bool = False
    continue_train: bool = False
    epoch_count: int = 1
    phase: str = "train"

    n_epochs: int = 1000
    n_epochs_decay: int = 100
    beta1: float = 0.5
    lr: float = 0.0002
    gan_mode: str = "lsgan"
    lambda_L1: float = 1.0
    pool_size: int = 50
    lr_policy: str = "linear"
    lr_decay_iters: int = 50
