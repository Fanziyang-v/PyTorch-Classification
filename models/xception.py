from torch import nn
from torch import Tensor


class Xception(nn.Module):
    """Xception."""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize Xception.

        Args:
            num_channels (int): number of channels of input feature map.
            num_classes (int): number of image classes.
        """
        super(Xception, self).__init__()
        self.model = nn.Sequential(
            # Entry flow.
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            XceptionModule1(64, 128, 128, relu=False),
            XceptionModule1(128, 256, 256),
            XceptionModule1(256, 728, 728),
            # Middle flow.
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            XceptionModule2(728),
            # Exit flow.
            XceptionModule1(728, 728, 1024),
            DepthwiseSeparableConv2d(1024, 1536, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(1536, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, images: Tensor):
        return self.model(images)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Convolution."""

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
            kernel_size (int): kernel size.
            stride (int, optional): stride of convolution. Defaults to 1.
            padding (int, optional): number of pixels for padding. Defaults to 0.
            dilation (int, optional): dilation rate. Defaults to 1.
        """
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class XceptionModule1(nn.Module):
    """Xception module in Entry or Exit flow."""

    def __init__(
        self, in_channels: int, mid_channels: int, out_channels: int, relu: bool = True
    ) -> None:
        """Initialize a Xception module in Entry or Exit flow.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            relu (bool, optional): use ReLU in the first layer. Defaults to True.
        """
        super(XceptionModule1, self).__init__()

        self.relu1 = nn.ReLU(inplace=True) if relu else nn.Identity()
        self.dsc1 = DepthwiseSeparableConv2d(
            in_channels, mid_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dsc2 = DepthwiseSeparableConv2d(
            mid_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x: Tensor):
        shortcut = self.conv(x)

        out = self.relu1(x)
        out = self.dsc1(out)
        out = self.bn1(out)

        out = self.relu2(out)
        out = self.dsc2(out)
        out = self.bn2(out)
        out = self.pool(out)

        out += shortcut
        return out


class XceptionModule2(nn.Module):
    """Xception Module in Middle flow."""

    def __init__(self, num_channels: int) -> None:
        """Initialize a Xception Module in Middle flow.

        Args:
            num_channels (int): number of channels of input and output feature map.
        """
        super(XceptionModule2, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.dsc1 = DepthwiseSeparableConv2d(
            num_channels, num_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.relu2 = nn.ReLU(inplace=True)
        self.dsc2 = DepthwiseSeparableConv2d(
            num_channels, num_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.relu3 = nn.ReLU(inplace=True)
        self.dsc3 = DepthwiseSeparableConv2d(
            num_channels, num_channels, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, x: Tensor):
        shortcut = x
        out = self.relu1(x)
        out = self.dsc1(out)
        out = self.bn1(out)

        out = self.relu2(out)
        out = self.dsc2(out)
        out = self.bn2(out)

        out = self.relu3(out)
        out = self.dsc3(out)
        out = self.bn3(out)

        out += shortcut
        return out
