"""
PyTorch implementation for ResNeXt.


For more details, see:
[1] Saining Xie et al.
    Aggregated Residual Transformations for Deep Neural Networks.
    http://arxiv.org/abs/1611.05431
"""

from torch import nn, Tensor


class ResNeXt(nn.Module):
    """ResNeXt.

    In this implementation, we replace the first 7x7 conv layer of stride 2 with a 3x3 conv layer of stride 1.
    Besides, we remove the first max pooling layer.
    """

    def __init__(
        self, num_classes: int, layers: list[int], channels: list[int], groups: int
    ) -> None:
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(channels[0], layers[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=2, groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x: Tensor):
        batch_size = x.size()[0]
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.avgpool(h)
        h = h.view(batch_size, -1)
        h = self.fc(h)
        return h

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
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(
                in_channels, out_channels * self.expansion, kernel_size=1, stride=stride
            )
            if stride != 1 or in_channels != out_channels * self.expansion
            else nn.Identity()
        )

    def forward(self, x: Tensor):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h += self.shortcut(x)
        h = self.relu3(h)
        return h


def resnext50(num_classes: int = 10, groups: int = 32) -> ResNeXt:
    return ResNeXt(num_classes, [3, 4, 6, 3], [128, 256, 512, 1024], groups=groups)


def resnext101(num_classes: int = 10, groups: int = 32) -> ResNeXt:
    return ResNeXt(num_classes, [3, 4, 23, 3], [128, 256, 512, 1024], groups=groups)


def resnext152(num_classes: int = 10, groups: int = 32) -> ResNeXt:
    return ResNeXt(num_classes, [3, 8, 36, 3], [128, 256, 512, 1024], groups=groups)
