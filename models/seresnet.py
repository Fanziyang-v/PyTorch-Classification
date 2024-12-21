"""
PyTorch implementation of SE-ResNet.

For more details, see:
[1] Jie Hu et al.
    Squeeze-and-Excitation Networks.
    http://arxiv.org/abs/1709.01507
"""

import torch
from torch import nn, Tensor


class SEResNet(nn.Module):
    """SEResNet.

    In this implementation, we replace the first 7x7 conv layer of stride 2 with a 3x3 conv layer of stride 1.
    Besides, we remove the first max pooling layer, resulting in a feature map of spatial size 32x32.
    """

    def __init__(
        self,
        num_classes: int,
        layers: list[int],
        channels: list[int],
        block: type["SEResidualBlock | SEBottleneck"],
    ) -> None:
        super(SEResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = _conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(channels[0], layers[0], block)
        self.layer2 = self._make_layer(channels[1], layers[1], block, stride=2)
        self.layer3 = self._make_layer(channels[2], layers[2], block, stride=2)
        self.layer4 = self._make_layer(channels[3], layers[3], block, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        block: type["SEResidualBlock | SEBottleneck"],
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample=downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SEResidualBlock(nn.Module):
    """SE-Residual Block."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
        downsample: nn.Module = None,
    ):
        super(SEResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    """SE-Bottleneck Block."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
        downsample: nn.Module = None,
    ):
        super(SEBottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels * self.expansion, reduction)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
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


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Construct a convolutional layer with kernel size of 3 and padding of 1.

    Args:
        in_channels (int): number of channels of input feature map.
        out_channels (int): number of channels of output feature map.
        stride (int, optional): stride of convolution. Defaults to 1.
    Returns:
        nn.Conv2d: 3x3 convolution layer with specific stride.
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        bias=False,
        padding=1,
    )


def seresnet18(num_classes: int) -> SEResNet:
    """SEResNet-18."""
    return SEResNet(num_classes, [2, 2, 2, 2], [64, 128, 256, 512], SEResidualBlock)


def seresnet34(num_classes: int) -> SEResNet:
    """SEResNet-34."""
    return SEResNet(num_classes, [3, 4, 6, 3], [64, 128, 256, 512], SEResidualBlock)


def seresnet50(num_classes: int) -> SEResNet:
    """SEResNet-50."""
    return SEResNet(num_classes, [3, 4, 6, 3], [64, 128, 256, 512], SEBottleneck)


def seresnet101(num_classes: int) -> SEResNet:
    """SEResNet-101."""
    return SEResNet(num_classes, [3, 4, 23, 3], [64, 128, 256, 512], SEBottleneck)


def seresnet152(num_classes: int) -> SEResNet:
    """SEResNet-152."""
    return SEResNet(num_classes, [3, 8, 36, 3], [64, 128, 256, 512], SEBottleneck)
