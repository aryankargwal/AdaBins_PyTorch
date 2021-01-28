import torch
import torch.nn as nn

from .layer_factory import PixelWiseDotProduct, PatchTransformer

from typing import Union


class MiniVIT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_query_channels: int = 128,
        patch_size: int = 16,
        out_dims: int = 256,
        embeddings_dims: int = 128,
        num_heads: int = 4,
        norm: str = "linear",
    ):
        super(MiniVIT, self).__init__()

        self.in_channels = in_channels
        self.num_query_channels = num_query_channels
        self.patch_size = patch_size
        self.out_dims = out_dims
        self.embeddings_dims = embeddings_dims
        self.num_heads = num_heads
        self.norm = norm

        if self.norm not in ["linear", "softmax", "sigmoid"]:
            raise ValueError(
                f"`norm` can be only `linear` or `softmax` but found {self.norm}"
            )

        self.conv3x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embeddings_dims,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.embeddings_dims, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.out_dims),
        )

        self.patch_transformer = PatchTransformer(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            embedding_dims=self.embeddings_dims,
            num_heads=self.num_heads,
        )
        self.dot_product = PixelWiseDotProduct()

    def forward(self, x: torch.Tensor):

        batch_size, channels, height, width = x.shape

        transformer_output = self.patch_transformer(x.clone())

        x = self.conv3x3(x)

        regression_head = transformer_output[0, ...]
        queries = transformer_output[1 : self.num_query_channels + 1, ...]

        # Now we have to convert from S, B, E to B, S, E
        # Check shapes once via loggin
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product(x, queries)

        reg_output = self.regressor(regression_head)
        if self.norm == "linear":
            reg_output = torch.relu(reg_output)
            eps = 0.1
            reg_output = reg_output + eps
        elif self.norm == "softmax":
            return torch.softmax(reg_output, dim=1), range_attention_maps
        else:
            reg_output = torch.sigmoid(reg_output)

        reg_output = reg_output / reg_output.sum(dim=1, keepdim=True)
        return reg_output, range_attention_maps
