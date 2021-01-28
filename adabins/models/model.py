import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniVIT import MiniVIT

import os

model_names = [
    "tf_efficientnet_b5_ra",
    "tf_efficientnet_b5_ap",
]

model_urls = {
    "tf_efficientnet_b5_ra": (
        "tf_efficientnet_b5_ra-9a3e5369.pth",
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth",  # noqa
    ),
    "tf_efficientnet_b5_ap": (
        "tf_efficientnet_b5_ap-9e82fae8.pth",
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth",  # noqa
    ),
}


class UpsampleBlock(nn.Module):
    def __init__(self, skip_input: int, out_channels: int):

        super(UpsampleBlock, self).__init__()

        self._net = nn.Sequential(
            nn.Conv2d(
                in_channels=skip_input,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor, concat_with: torch.Tensor):

        upsampled_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        y = torch.cat([upsampled_x, concat_with], dim=1)

        y = self._net(y)

        return y


class DecoderBlock(nn.Module):
    def __init__(
        self,
        num_features: int = 2048,
        num_classes: int = 1,
        bottleneck_features: int = 2048,
    ):
        super(DecoderBlock, self).__init__()

        features = int(num_features)

        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_features,
            out_channels=features,
            kernel_size=1,
            stride=1,
            padding=1,
        )

        self.upsample1 = UpsampleBlock(
            skip_input=features // 1 + 112 + 64, out_channels=features // 2
        )
        self.upsample2 = UpsampleBlock(
            skip_input=features // 2 + 40 + 24, out_channels=features // 4
        )
        self.upsample3 = UpsampleBlock(
            skip_input=features // 4 + 24 + 16, out_channels=features // 8
        )
        self.upsample4 = UpsampleBlock(
            skip_input=features // 8 + 16 + 8, out_channels=features // 16
        )

        self.conv3 = nn.Conv2d(
            in_channels=features // 16,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, features: torch.Tensor):

        x0, x1, x2, x3, x4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )

        xd0 = self.conv2(x4)

        xd1 = self.upsample1(xd0, x3)
        xd2 = self.upsample2(xd1, x2)
        xd3 = self.upsample3(xd2, x1)
        xd4 = self.upsample4(xd3, x0)

        res = self.conv3(xd4)

        return res


class EncoderBlock(nn.Module):
    def __init__(self, backend):
        super(EncoderBlock, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for key, value in self.original_model._modules.items():
            if key == "blocks":
                for ki, vi in value._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(value(features[-1]))

        return features


class Adabins(nn.Module):
    def __init__(
        self,
        backend,
        num_bins: int = 100,
        min_val: int = 0.1,
        max_val: int = 10,
        norm: str = "linear",
    ):
        super(Adabins, self).__init__()

        print(type(backend))

        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.norm = norm

        self.encoder = EncoderBlock(backend=backend)

        self.adaptive_bins_layer = MiniVIT(
            in_channels=128,
            num_query_channels=128,
            patch_size=16,
            out_dims=num_bins,
            embeddings_dims=128,
            norm=norm,
        )

        self.decoder = DecoderBlock(num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=num_bins,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x, **kwargs):

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output, **kwargs)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(
            decoder_output
        )

        output = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode="constant", value=self.min_val
        )
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centres = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centres.size()
        centres = centres.view(n, dout, 1, 1)

        preds = torch.sum(output * centres, dim=1, keepdim=True)

        return bin_edges, preds


def build_adabins(num_bins: int, pretrained: bool = True, **kwargs):

    backbone = "tf_efficientnet_b5_ap"

    print(f"Loading Backbone {backbone}....")

    if os.path.exists(
        f"/home/pranjal/.cache/torch/hub/checkpoints/{model_urls[backbone][0]}"
    ):
        print(
            f"/home/pranjal/.cache/torch/hub/checkpoints/{model_urls[backbone][0]} found!"
        )
        backbone = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            backbone,
            pretrained=True,
            verbose=True,
        )
    else:

        backbone = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            backbone,
            pretrained=False,
            verbose=True,
        )

    print("Backbone loaded.")

    print(f"Removing last two layers : global-pool dand classifier")

    backbone.global_pool = nn.Identity()
    backbone.classifier = nn.Identity()

    print(f"Instantiating Adabins Model...")
    model = Adabins(backend=backbone, num_bins=num_bins, **kwargs)
    print("Instantiated.")

    return model


if __name__ == "__main__":
    model = build_adabins(100)

    x = torch.rand(2, 3, 480, 640)
    bins, preds = model(x)
    print(f"bins: {bins}\npreds: {preds}")
