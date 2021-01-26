# Library Imports
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import random
from sklearn.utils import shuffle
from zipfile import ZipFile


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# Loading ZIP data
def loadZipToMem(zip_file):
    print("Loading dataset zip file...", end="")

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (
            row.split(",")
            for row in (data["data/nyu2_train.csv"]).decode("utf-8").split("\n")
            if len(row) > 0
        )
    )

    nyu2_train = shuffle(nyu2_train, random_state=0)
    print("Loaded ({0}).".format(len(nyu2_train)))
    return data, nyu2_train


# Getting items from the Zip
class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


# Converting the NumPy Array to a Torch Tensor
class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        image, focal = sample["image"], sample["focal"]
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == "test":
            return {"image": image, "focal": focal}

        depth = sample["depth"]
        if self.mode == "train":
            depth = self.to_tensor(depth)
            return {"image": image, "depth": depth, "focal": focal}
        else:
            has_valid_depth = sample["has_valid_depth"]
            return {
                "image": image,
                "depth": depth,
                "focal": focal,
                "has_valid_depth": has_valid_depth,
                "image_path": sample["image_path"],
                "depth_path": sample["depth_path"],
            }

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


# Pre-Processing
def rotate_image(self, image, angle, flag=Image.BILINEAR):
    result = image.rotate(angle, resample=flag)
    return result


def random_crop(self, img, depth, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == depth.shape[0]
    assert img.shape[1] == depth.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y : y + height, x : x + width, :]
    depth = depth[y : y + height, x : x + width, :]
    return img, depth


def image_augment(self, image):
    # Gamma Augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # Color Augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)

    return image_aug


# !!YET TO ADD: CODE TO TRANSFORM THE DATA!!
