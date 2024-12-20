"""
PyTorch implementation for MobileNetV2.

In this implementation, we don't use width and resolution multiplier hyperparameters.

For more details, see:
[1] Mark Sandler et al.
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    http://arxiv.org/abs/1801.04381
"""

from torch import nn, Tensor


class MobileNetV2(nn.Module):
    """MobileNetV2"""

    def __init__(self, num_classes: int) -> None:
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU6(inplace=True)
        # Bottlenecks
        self.layer1 = self._make_layer(32, 16, num_blocks=1, expansion=1)
        self.layer2 = self._make_layer(16, 24, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(24, 32, num_blocks=3, stride=2)
        self.layer4 = self._make_layer(32, 64, num_blocks=4, stride=2)
        self.layer5 = self._make_layer(64, 96, num_blocks=3)
        self.layer6 = self._make_layer(96, 160, num_blocks=3, stride=2)
        self.layer7 = self._make_layer(160, 320, num_blocks=1)
        self.conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1280)
        self.relu = nn.ReLU6(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(1280, num_classes, kernel_size=1, bias=False)
        self._init_weights()

    def forward(self, images: Tensor):
        out = self.conv1(images)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.fc(out)
        out = out.view(images.size(0), -1)
        return out

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        expansion: int = 6,
    ) -> nn.Sequential:
        layer = [Bottleneck(in_channels, out_channels, stride, expansion)]
        for _ in range(num_blocks - 1):
            layer.append(Bottleneck(out_channels, out_channels, expansion=expansion))
        return nn.Sequential(*layer)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)


class Bottleneck(nn.Module):
    """Bottleneck in MobileNetV2.

    Architecture: [1x1 conv - bn - relu6] - [3x3 depthwise conv - bn - relu6] - [1x1 conv - bn]
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 6
    ) -> None:
        """Initialize a Bottleneck.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            stride (int, optional): stride of depthwise convolution. Defaults to 1.
            expansion (int, optional): expansion ratio for the first 1x1 convolution. Defaults to 6.
        """
        super(Bottleneck, self).__init__()
        self.shortcut = stride == 1 and in_channels == out_channels
        mid_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU6(inplace=True)
        # depthwise convolution
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU6(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.shortcut:
            out += x
        return out
