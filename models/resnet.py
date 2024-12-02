"""
PyTorch implementation for ResNet.

For more details, see: 
[1] Kaiming He et al. 
    Deep Residual Learning for Image Recognition. 
    http://arxiv.org/abs/1512.03385
"""

from torch import nn, Tensor


class ResNet(nn.Module):
    """ResNet"""

    def __init__(
        self,
        num_classes: int,
        layers: list[int],
        channels: list[int],
        block: type["Bottleneck | ResidualBlock"],
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        # the first 7x7 conv with stride of 2 is replaced by three 3x3 conv
        self.conv1 = _conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = _conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channels[0], layers[0], block)
        self.layer2 = self._make_layer(channels[1], layers[1], block, stride=2)
        self.layer3 = self._make_layer(channels[2], layers[2], block, stride=2)
        self.layer4 = self._make_layer(channels[3], layers[3], block, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)
        self._init_weights()

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        block: type["ResidualBlock | Bottleneck"],
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
        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample=downsample,
            )
        )
        self.in_channels = out_channels * block.expansion

        for _ in range(num_blocks - 1):
            layers.append(block(out_channels * block.expansion, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    """Basic Residual Block in ResNet."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initialize a basic residual block.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            stride(int): stride of the first convolution. Defaults to 1.
            downsample (nn.Module | None, optional): downsampling module. Defaults to None.
        """
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck Block in ResNet."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initialize a bottleneck block in ResNet.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of middle feature map.
            stride (int, optional): stride of convolution. Defaults to 1.
            downsample (nn.Module | None, optional): downsampling module. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = _conv3x3(out_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x: Tensor):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu3(out)
        return out


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


def resnet18(num_classes: int = 10) -> ResNet:
    return ResNet(num_classes, [2, 2, 2, 2], [64, 128, 256, 512], ResidualBlock)


def resnet34(num_classes: int = 10) -> ResNet:
    return ResNet(num_classes, [3, 4, 6, 3], [64, 128, 256, 512], ResidualBlock)


def resnet50(num_classes: int = 10) -> ResNet:
    return ResNet(num_classes, [3, 4, 6, 3], [64, 128, 256, 512], Bottleneck)


def resnet101(num_classes: int = 10) -> ResNet:
    return ResNet(num_classes, [3, 4, 23, 3], [64, 128, 256, 512], Bottleneck)


def resnet152(num_classes: int = 10) -> ResNet:
    return ResNet(num_classes, [3, 8, 36, 3], [64, 128, 256, 512], Bottleneck)
