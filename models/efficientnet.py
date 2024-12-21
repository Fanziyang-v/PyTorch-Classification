"""
PyTorch implementation for EfficientNet.

For more details, see:
[1] Mingxing Tan et al.
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    http://arxiv.org/abs/1905.11946
"""

import torch
from torch import nn, Tensor


class EfficientNet(nn.Module):
    """EfficientNet.

    In this implementation, we replace the first 3x3 conv layer of stride 2 with a 3x3 conv layer of stride 1.
    Besides, we replace the 3x3 conv layer of stride 2 in the first conv layer of stride 1.
    """

    def __init__(self, num_classes: int) -> None:
        super(EfficientNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.swish = nn.SiLU(inplace=True)
        # MBConv blocks
        self.layer1 = self._make_layer(32, 16, num_blocks=1, kernel_size=3, expansion=1)
        self.layer2 = self._make_layer(16, 24, num_blocks=2, kernel_size=3)
        self.layer3 = self._make_layer(24, 40, num_blocks=2, stride=2, kernel_size=5)
        self.layer4 = self._make_layer(40, 80, num_blocks=3, stride=2, kernel_size=3)
        self.layer5 = self._make_layer(80, 112, num_blocks=3, kernel_size=5)
        self.layer6 = self._make_layer(112, 192, num_blocks=4, stride=2, kernel_size=5)
        self.layer7 = self._make_layer(192, 320, num_blocks=1, kernel_size=3)
        self.conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)
        self._init_weights()

    def forward(self, images: Tensor):
        out = self.conv1(images)
        out = self.bn1(out)
        out = self.swish(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: int = 6,
    ) -> nn.Sequential:
        layer = [MBConv(in_channels, out_channels, kernel_size, stride, expansion)]
        for _ in range(num_blocks - 1):
            layer.append(
                MBConv(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                )
            )
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


class MBConv(nn.Module):
    """MBConv block in EfficientNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: int = 6,
    ) -> None:
        super(MBConv, self).__init__()
        self.shortcut = stride == 1 and in_channels == out_channels
        mid_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=mid_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU(inplace=True)
        self.se = SEBlock(mid_channels)

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.swish(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.shortcut:
            out += x
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.squeeze(x)
        out = self.excitate(out)
        return x * out


if __name__ == "__main__":
    model = EfficientNet(num_classes=10)
    print(model)
    images = torch.randn(1, 3, 32, 32)
    outputs = model(images)
    print(outputs.shape)
