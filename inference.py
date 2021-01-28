import cv2
from PIL import Image
import os
import torch
from typing import Union
import numpy as np
import torch.nn as nn
from adabins.models import build_adabins
from adabins.models.utils import load_checkpoint
import gdown

file_id = "1bKIg56NyxQdXNWS1jfhZuatxD_VE0seN"


class AdabinsInference(object):
    def __init__(
        self,
        num_bins: int,
        model_path: Union[str, None] = None,
        min_val: int = 1e-3,
        max_val: int = 80,
        device: str = "cuda",
        norm: str = "linear",
        pretrained: bool = True,
    ):
        self.num_bins = num_bins
        self.min_depth = min_val
        self.max_depth = max_val
        self.device = device
        self.norm = norm
        self.pretrained = pretrained
        self.model_path = model_path

        if model_path is None:
            self.download_weights()

        self.model = build_adabins(
            num_bins=self.num_bins,
            min_val=self.min_depth,
            max_val=self.max_depth,
            pretrained=self.pretrained,
        )

        self.model, _, _ = load_checkpoint(self.model_path, self.model)

        self.model = self.model.eval()

        self.model = self.model.to(self.device)

    def download_weights(self):

        print("Downloading weights ...")
        if os.path.exists("weights/"):
            pass
        else:
            os.mkdir("weights/")
        self.model_path = "weights/AdaBins_nyu.pt"
        url = "https://drive.google.com/uc?id={}".format(file_id)
        gdown.download(url=url, output=self.model_path, quiet=False)

    def predict(
        self,
        img: Union[np.ndarray, Image.Image, str],
        show: bool = False,
        cmap: str = "plasma",
    ) -> np.ndarray:

        if isinstance(img, str):
            img = cv2.imread(img)
            img = img / 255.0
            img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0)
        elif isinstance(img, np.ndarray):
            img = img / 255.0
            img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0)
        elif isinstance(img, Image.Image):

            # handle PIL Image
            if img.mode == "I":
                img = torch.from_numpy(np.array(img, np.int32, copy=False))
            elif img.mode == "I;16":
                img = torch.from_numpy(np.array(img, np.int16, copy=False))
            else:
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if img.mode == "YCbCr":
                nchannel = 3
            elif img.mode == "I;16":
                nchannel = 1
            else:
                nchannel = len(img.mode)
            img = img.view(img.size[1], img.size[0], nchannel)

            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img, torch.ByteTensor):
                img = img.float()

        # Actual inference begins
        with torch.no_grad():
            bins, pred = self.model(img.to(self.device))
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        img = torch.Tensor(np.array(img.cpu().numpy())[..., ::-1].copy()).to(
            self.device
        )
        with torch.no_grad():
            pred_lr = self.model(img)[-1]
        pred_lr = np.clip(
            pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth
        )

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = (
            nn.functional.interpolate(
                torch.Tensor(final),
                img.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            .cpu()
            .numpy()
        )

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        if show:
            plt.imshow(final.squeeze(), cmap=cmap)
            plt.show()

        return centers, final.squeeze()
