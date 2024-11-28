"""
PyTorch implementation for GoogLeNet.

For more details: see: 
[1] Christian Szegedy et.al Going Deeper with Convolutions. http://arxiv.org/abs/1409.4842

"""

import torch
from torch import nn
from torch import Tensor


class GoogLeNet(nn.Module):
    """GoogLeNet."""

    def __init__(
        self, num_channels: int, num_classes: int, use_aux: bool = False
    ) -> None:
        """Initialize GoogLeNet.

        Args:
            num_channels(int): number of channels of input images.
            num_classes(int): number of classes.
            use_aux(bool): use auxiliary classifiers if use_aux is True.
        """
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.inception3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.inception4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.inception4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.inception4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.inception4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.inception5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1024, num_classes)
        if use_aux:
            # auxiliary classifier 1 is applied after inception4_1
            self.aux1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            )
            # auxiliary classifier 2 is applied after inception4_4
            self.aux2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),
                nn.Conv2d(528, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            )
        else:
            self.aux1 = nn.Identity()
            self.aux2 = nn.Identity()

    def forward(self, images: Tensor):
        out = self.conv1(images)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.inception3_1(out)
        out = self.inception3_2(out)
        out = self.pool3(out)

        out = self.inception4_1(out)
        # apply auxiliary classifier 1
        out_aux1 = self.aux1(out)
        out = self.inception4_2(out)
        out = self.inception4_3(out)
        out = self.inception4_4(out)
        # apply auxiliary classifier 2
        out_aux2 = self.aux2(out)
        out = self.inception4_5(out)
        out = self.pool4(out)

        out = self.inception5_1(out)
        out = self.inception5_2(out)
        out = self.pool5(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, out_aux1, out_aux2


class Inception(nn.Module):
    """Inception Module in GoogLeNet."""

    def __init__(
        self,
        in_channels: int,
        c1: int,
        c2: tuple[int, int],
        c3: tuple[int, int],
        c4: int,
    ) -> None:
        """Initialize an Inception Module."""
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1, stride=1)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1, stride=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1, stride=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, stride=1, padding=2)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1, stride=1)
        self.relu4_2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass in Inception Module."""
        out1 = self.relu1_1(self.p1_1(x))
        out2 = self.relu2_2(self.p2_2(self.relu2_1(self.p2_1(x))))
        out3 = self.relu3_2(self.p3_2(self.relu3_1(self.p3_1(x))))
        out4 = self.relu4_2(self.p4_2(self.p4_1(x)))
        return torch.cat([out4, out3, out2, out1], dim=1)
