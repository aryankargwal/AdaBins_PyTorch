import torch
import torch.nn as nn
import numpy as np


class PatchTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 10,
        embedding_dims: int = 128,
        num_heads: int = 4,
        transformer_feedforward: int = 1024,
        transformer_num_layers: int = 4,
    ):

        super(PatchTransformer, self).__init__()

        self.embedding_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dims,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dims,
            nhead=num_heads,
            dim_feedforward=transformer_feedforward,
        )

        # Takes input of shape B x S x E
        # B : Batch Size
        # E : Patch Embeddings Dim
        # S : (h * w) / p * p, where h, w are height and width of the decoded features
        self.transformerEncoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer, num_layers=transformer_num_layers
        )

        self.positional_encodings = nn.Parameter(
            torch.rand(500, embedding_dims), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patch_encodings = self.embedding_conv(x).flatten(
            start_dim=2
        )  # (B, C, H, W) -> (B, C, H*W)

        patch_encodings = patch_encodings + self.positional_encodings[
            : patch_encodings.shape[2], :
        ].T.unsqueeze(0)

        # Conver to train [B, C, S] as expected by the transformer
        patch_encodings = patch_encodings.permute(2, 0, 1)

        x = self.transformerEncoder(patch_encodings)

        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(
        self, x: torch.Tensor, transformer_output: torch.Tensor
    ) -> torch.Tensor:

        batch_size, channels, height, width = x.shape
        _, out_channels, hw = transformer_output.shape

        # Through assertion error if x and transformer_output have different channel depths
        assert (
            channels == out_channels
        ), f"Number of channels in `x` and `transformer_output` need to be same"

        prod = torch.matmul(
            x.view(batch_size, channels, height * width).permute(0, 2, 1),
            transformer_output.permute(0, 2, 1),
        )

        return prod.permute(0, 2, 1).view(batch_size, out_channels, height, width)
