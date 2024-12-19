"""
PyTorch implementation for ShuffleNet.

For more details: see:
[1] Zhang, Xiangyu, Xinyu Zhou, Mengxiao Lin, and Jian Sun. 
    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices. 
    https://doi.org/10.48550/arXiv.1707.01083.

"""

import torch
from torch import nn, Tensor


class ShuffleNet(nn.Module):
    """ShuffleNet model.

    In this implementation, we replace the first 3x3 conv layer of stride 2 with a 3x3 conv layer of stride 1.
    Besides, we remove the first 3x3 max pooling layer, resulting in a feature map of spatial size 32x32.
    """

    cfg = {
        1: [24, 36, 72, 144],
        2: [24, 50, 100, 200],
        3: [24, 60, 120, 240],
        4: [24, 68, 138, 276],
        8: [24, 96, 192, 384],
    }

    def __init__(
        self, num_classes: int, groups: int = 3, width_mult: float = 1.0
    ) -> None:
        super(ShuffleNet, self).__init__()
        if groups not in self.cfg:
            raise ValueError(f"Unsupported groups: {groups}")
        layers = self.cfg[groups]
        self.in_planes = int(layers[0] * width_mult)
        self.conv1 = nn.Conv2d(
            3,
            self.in_planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.stage2 = self._make_stage(
            in_planes=self.in_planes,
            planes=int(layers[1] * width_mult),
            num_units=4,
            stride=2,
            groups=groups,
            first_group_conv=False,
        )
        self.stage3 = self._make_stage(
            in_planes=self.in_planes,
            planes=int(layers[2] * width_mult),
            num_units=8,
            stride=2,
            groups=groups,
        )
        self.stage4 = self._make_stage(
            in_planes=self.in_planes,
            planes=int(layers[3] * width_mult),
            num_units=4,
            stride=2,
            groups=groups,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            int(layers[3] * width_mult) * ShuffleNetUnit.expansion, num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _make_stage(
        self,
        in_planes: int,
        planes: int,
        num_units: int,
        stride: int,
        groups: int,
        first_group_conv: bool = True,
    ) -> nn.Sequential:
        self.in_planes = planes * ShuffleNetUnit.expansion
        layers = [
            ShuffleNetUnit(
                in_planes,
                planes,
                stride=stride,
                groups=groups,
                first_group_conv=first_group_conv,
            )
        ]
        for _ in range(num_units - 1):
            layers.append(
                ShuffleNetUnit(
                    planes * ShuffleNetUnit.expansion, planes, stride=1, groups=groups
                )
            )
        return nn.Sequential(*layers)


class ShuffleNetUnit(nn.Module):
    """ShuffleNet unit."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        groups: int,
        first_group_conv: bool = True,
    ) -> None:
        super(ShuffleNetUnit, self).__init__()
        if stride not in [1, 2]:
            raise ValueError(f"Unsupported stride: {stride}")
        self.stride = stride
        self.gconv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=1,
            groups=groups if first_group_conv else 1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.depthwise_conv = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.gconv2 = nn.Conv2d(
            planes,
            planes * self.expansion - (in_planes if stride == 2 else 0),
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion - (in_planes if stride == 2 else 0)
        )
        self.relu = nn.ReLU(inplace=True)
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = _shuffle_channels(out, groups=1)
        out = self.depthwise_conv(out)
        out = self.bn2(out)
        out = self.gconv2(out)
        out = self.bn3(out)
        if self.stride == 2:
            out = torch.cat((out, self.shortcut(x)), dim=1)
        else:
            out += x
        out = self.relu(out)
        return out


def _shuffle_channels(x: Tensor, groups: int) -> Tensor:
    """Performs channel shuffle operation.

    Args:
        x (Tensor): input feature map.
        groups (int): groups to split channels.

    Returns:
        Tensor: feature map after channel shuffle.
    """
    B, C, H, W = x.size()
    n = C // groups
    # Reshape and transpose x
    x = x.view(B, groups, n, H, W).transpose(1, 2).contiguous()
    # Flatten x
    x = x.view(B, C, H, W)
    return x


def shufflenet(
    num_classes: int = 10, groups: int = 3, width_mult: float = 1.0
) -> ShuffleNet:
    """Construct a ShuffleNet model.

    Args:
        num_classes (int, optional): number of classes. Defaults to 10.
        groups (int, optional): number of groups for group convolution. Defaults to 3.
        width_mult (float, optional): width multiplier to control the number of filters in each layer. Defaults to 1.0.

    Returns:
        ShuffleNet: shuffled network model.
    """
    return ShuffleNet(num_classes, groups=groups, width_mult=width_mult)
