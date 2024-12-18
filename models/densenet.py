"""
PyTorch implementation for DenseNet.

For more details, see:
[1] Gao Huang, Zhuang Liu et al.
    Densely Connected Convolutional Networks.
    https://arxiv.org/abs/1608.06993v5
"""

import torch
from torch import nn, Tensor


class DenseNet(nn.Module):
    """DenseNet.

    In this implementation, we replace the first 7x7 conv of stride 2 with 3x3 conv with stride 1.
    Besides, we remove the first 3x3 max pooling of stride 2.
    """

    def __init__(
        self,
        num_classes: int,
        layers: list[int],
        growth_rate: int = 32,
        theta: float = 0.5,
    ):
        super(DenseNet, self).__init__()
        num_features = 2 * growth_rate
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            self.layers.append(DenseBlock(layers[i], num_features, growth_rate))
            num_features += layers[i] * growth_rate
            if i < len(layers):
                self.layers.append(
                    TransitionLayer(num_features, int(num_features * theta))
                )
                num_features = int(num_features * theta)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.bn1(x)
        h = self.relu(h)
        h = self.conv1(h)
        for layer in self.layers:
            h = layer(h)
        h = self.avgpool(h)
        h = torch.flatten(h, start_dim=1)
        h = self.fc(h)
        return h


class DenseLayer(nn.Module):
    """Dense Layer"""

    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.bn1(x)
        h = self.relu(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.conv2(h)
        return torch.cat([x, h], dim=1)


class DenseBlock(nn.Module):
    """Dense Block"""

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int) -> None:
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                DenseLayer(in_channels + i * growth_rate, growth_rate)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class TransitionLayer(nn.Module):
    """Transition Layer"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        h = self.bn(x)
        h = self.relu(h)
        h = self.conv(h)
        h = self.avgpool(h)
        return h


def densenet121(num_classes: int) -> DenseNet:
    return DenseNet(num_classes=num_classes, layers=[6, 12, 24, 16])


def densenet169(num_classes: int) -> DenseNet:
    return DenseNet(num_classes=num_classes, layers=[6, 12, 32, 32])


def densenet201(num_classes: int) -> DenseNet:
    return DenseNet(num_classes=num_classes, layers=[6, 12, 48, 32])


def densenet264(num_classes: int) -> DenseNet:
    return DenseNet(num_classes=num_classes, layers=[6, 12, 64, 48])
