"""
PyTorch implementation for MobileNetV1.

For more details, see:
[1] Andrew G. Howard et al.
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
    http://arxiv.org/abs/1704.04861
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math


class MobileNetV1(nn.Module):
    """MobileNet V1."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        """Initialize a MobileNetV1.

        Args:
            num_classes (int): number of classes of images.
            alpha (float, optional): width multiplier. Defaults to 1.0.
            beta (float, optional): resolution mutiplier. Defaults to 1.0.
        """
        super(MobileNetV1, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.model = nn.Sequential(
            nn.Conv2d(3, math.ceil(32 * alpha), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(math.ceil(32 * alpha)),
            nn.ReLU(inplace=True),
            DepthwiseSaparableConv2d(
                math.ceil(32 * alpha), math.ceil(64 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(64 * alpha),
                math.ceil(128 * alpha),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            DepthwiseSaparableConv2d(
                math.ceil(128 * alpha), math.ceil(128 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(128 * alpha),
                math.ceil(256 * alpha),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            DepthwiseSaparableConv2d(
                math.ceil(256 * alpha), math.ceil(256 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(256 * alpha),
                math.ceil(512 * alpha),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            DepthwiseSaparableConv2d(
                math.ceil(512 * alpha), math.ceil(512 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(512 * alpha), math.ceil(512 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(512 * alpha), math.ceil(512 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(512 * alpha), math.ceil(512 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(512 * alpha), math.ceil(512 * alpha), kernel_size=3, padding=1
            ),
            DepthwiseSaparableConv2d(
                math.ceil(512 * alpha),
                math.ceil(1024 * alpha),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(math.ceil(1024 * alpha), num_classes),
        )

    def forward(self, images: Tensor):
        if self.beta < 1.0:
            _, _, h, w = images.size()
            h, w = torch.ceil(torch.tensor([h, w]) * self.beta).int()
            images = F.interpolate(images, size=(h, w), mode="bilinear")
        return self.model(images)


class DepthwiseSaparableConv2d(nn.Module):
    """Depthwise separable convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super(DepthwiseSaparableConv2d, self).__init__()
        # depthwise convolution
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # pointwise convolution
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.pw_conv(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out
