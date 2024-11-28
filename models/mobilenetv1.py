"""
PyTorch implementation for MobileNetV1.

For more details, see:
[1] Andrew G. Howard et.al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
    http://arxiv.org/abs/1704.04861
"""

from torch import nn, Tensor


class MobileNetV1(nn.Module):
    """MobileNet V1."""

    def __init__(self, num_channles: int, num_classes: int) -> None:
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(num_channles, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.model = nn.Sequential(
            nn.Conv2d(num_channles, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSaparableConv2d(32, 64, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(64, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseSaparableConv2d(128, 128, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(128, 256, kernel_size=3, stride=2, padding=1),
            DepthwiseSaparableConv2d(256, 256, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(256, 512, kernel_size=3, stride=2, padding=1),
            DepthwiseSaparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthwiseSaparableConv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, images: Tensor):
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
        """Initialize a depthwise separable convolution layer.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            kernel_size (int, optional): kernel size.
            stride (int, optional): stride of convolution. Defaults to 1.
            padding (int, optional): number of pixels padding. Defaults to 0.
            dilation (int, optional): dilation rate. Defaults to 1.
        """
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
        self.pw_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
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
