from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dih.blocks import ConvBlock, DeconvBlock


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.005)
        nn.init.ones_(m.bias)


class SharedEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            ConvBlock(4, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 512)
        ])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 1024)
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for layer in self.conv_layers:
            x = layer(x)
            skips.append(x)

        z = self.linear(x)

        return z, skips


class SceneParsingDecoder(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.deconv_layers = nn.ModuleList([
            DeconvBlock(1024, 512, padding=0),
            DeconvBlock(512, 256),
            DeconvBlock(256, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 64),

        ])
        self.output_layers = nn.Sequential(
            DeconvBlock(64, 64),
            ConvBlock(
                64,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=False,
                bn=False
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        feature_maps = []
        for layer, skip in zip(self.deconv_layers, reversed(skips)):
            x = layer(x)
            x = x + skip
            feature_maps.append(x)

        mask = self.output_layers(x)

        return mask, feature_maps


class HarmonizationDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.deconv_layers = nn.ModuleList([
            DeconvBlock(1024, 512, padding=0),
            DeconvBlock(512*2, 256),
            DeconvBlock(256*2, 256),
            DeconvBlock(256*2, 128),
            DeconvBlock(128*2, 128),
            DeconvBlock(128*2, 64),
            DeconvBlock(64*2, 64),
        ])
        self.output_layer = DeconvBlock(64, 3, activation=False, bn=False)

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        feature_maps: List[torch.Tensor]
    ) -> torch.Tensor:

        # don't use last feature map
        feature_maps[-1] = torch.tensor([])

        for layer, skip, feat_map in zip(
            self.deconv_layers, reversed(skips), feature_maps
        ):
            x = layer(x)
            x = x + skip
            x = torch.cat((x, feat_map), dim=1)

        x = self.output_layer(x)

        return x


class DeepImageHarmonization(nn.Module):

    def __init__(self, num_classes: int = 25):
        super().__init__()
        self.encoder = SharedEncoder()
        self.scene_decoder = SceneParsingDecoder(num_classes)
        self.decoder = HarmonizationDecoder()

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Concatenate image with mask
        x = torch.cat((x, mask), dim=1)

        # Encode image
        z, skips = self.encoder(x)
        z = z.reshape(x.shape[0], -1, 1, 1)

        # Decode to segmentation mask
        mask, feature_maps = self.scene_decoder(z, skips)

        # Decode to harmonized image
        y_pred = self.decoder(z, skips, feature_maps)

        return y_pred, mask
