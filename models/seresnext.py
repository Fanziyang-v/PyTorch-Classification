"""
PyTorch implementation for SE-ResNeXt.


For more details, see:
[1] Jie Hu et al.
    Squeeze-and-Excitation Networks.
    http://arxiv.org/abs/1709.01507
"""

import torch
from torch import nn, Tensor


class SEResNeXt(nn.Module):
    """SE-ResNeXt.

    In this implementation, we replace the first 7x7 conv layer of stride 2 with a 3x3 conv layer of stride 1.
    Besides, we remove the first max pooling layer.
    """

    def __init__(
        self, num_classes: int, layers: list[int], channels: list[int], groups: int
    ) -> None:
        super(SEResNeXt, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(channels[0], layers[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=2, groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int = 1, groups: int = 32
    ) -> nn.Sequential:
        layers = [
            Bottleneck(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                groups=groups,
            )
        ]
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(num_blocks - 1):
            layers.append(
                Bottleneck(
                    in_channels=out_channels * Bottleneck.expansion,
                    out_channels=out_channels,
                    stride=1,
                    groups=groups,
                )
            )
        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    """Bottleneck block in ResNeXt."""

    expansion = 2

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 32
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels * self.expansion)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
            if stride != 1 or in_channels != out_channels * self.expansion
            else nn.Identity()
        )

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.squeeze(x)
        out = self.excitate(out)
        return x * out


def seresnext50(num_classes: int = 10, groups: int = 32) -> SEResNeXt:
    return SEResNeXt(num_classes, [3, 4, 6, 3], [128, 256, 512, 1024], groups=groups)


def seresnext101(num_classes: int = 10, groups: int = 32) -> SEResNeXt:
    return SEResNeXt(num_classes, [3, 4, 23, 3], [128, 256, 512, 1024], groups=groups)


def seresnext152(num_classes: int = 10, groups: int = 32) -> SEResNeXt:
    return SEResNeXt(num_classes, [3, 8, 36, 3], [128, 256, 512, 1024], groups=groups)
