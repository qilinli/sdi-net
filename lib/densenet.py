from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        growth_rate: int,
        depth: int,
        kernel_size: tuple[int, int],
        activation: Callable[[], nn.Module],
        bn_params: dict,
    ) -> None:
        super(DenseBlock, self).__init__()
        layers = []
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        for i in range(depth):
            layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate, **bn_params),
                    activation(),
                    nn.Conv2d(
                        in_channels + i * growth_rate,
                        latent_channels,
                        kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(latent_channels, **bn_params),
                    activation(),
                    nn.Conv2d(latent_channels, growth_rate, 1),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [x]
        for layer in self.layers:
            xs.append(layer(torch.cat(xs, dim=1)))
        return torch.cat(xs, dim=1)


class SDIDenseNet(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        growth_rate: int = 12,
        structure: Sequence[int] = (6, 6, 6),
        patch_size: int = 20,
        patch_stride: int = 5,
        compression: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        kernel_size: tuple[int, int] = (5, 1),
        bn_params: dict = {"momentum": 0.01, "eps": 1.1e-5},
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                base_channels,
                (patch_size, 1),
                (patch_stride, 1),
                (patch_size // 2, 0),
                bias=False,
            )
        ]
        channels = base_channels
        latent_channels = 2 * growth_rate
        for i, depth in enumerate(structure):
            if i != 0:
                layers.append(nn.BatchNorm2d(channels, **bn_params))
                layers.append(nn.AvgPool2d((2, 1)))
                layers.append(activation())
                new_channels = round(channels * (1.0 - compression))
                layers.append(nn.Conv2d(channels, new_channels, 1, bias=False))
                channels = new_channels
            layers.append(
                DenseBlock(
                    channels,
                    latent_channels,
                    growth_rate,
                    depth,
                    kernel_size,
                    activation,
                    bn_params,
                )
            )
            channels += growth_rate * depth
        super(SDIDenseNet, self).__init__(*layers)
