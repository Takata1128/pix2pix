from PIL import Image
import os
from typing import Optional
from torchvision import transforms
from PIL import Image
import numpy as np
from config import Config
from torchvision import transforms

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


def get_params(config: Config, size: int):
    w, h = size
    new_h = h
    new_w = w

    if "resize" in config.preprocess:
        new_h = new_w = config.load_size
    elif "scale_width" in config.preprocess:
        new_w = config.load_size
        new_h = config.load_size * h // w

    x = np.random.randint(0, np.maximum(0, new_w - config.crop_size))
    y = np.random.randint(0, np.maximum(0, new_h - config.crop_size))

    flip = np.random.rand() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(
    config: Config,
    params: Optional[dict] = None,
    grayscale: bool = False,
    method=transforms.InterpolationMode.BICUBIC,
    convert: bool = True,
):
    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if "resize" in config.preprocess:
        osize = [config.load_size, config.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in config.preprocess:
        transform_list.append(
            transforms.Lambda(
                lambda img: __scale_width(
                    img, config.load_size, config.crop_size, method
                )
            )
        )

    if "crop" in config.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(config.crop_size))
        else:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __crop(img, params["crop_pos"], config.crop_size)
                )
            )

    if config.preprocess == "none":
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
        )

    if not config.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params["flip"]))
            )

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(
    img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC
):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True
