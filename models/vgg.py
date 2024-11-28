"""
PyTorch implementation for VGGNet.

For more details, see:
[1] Karen Simonyan et.al. Very Deep Convolutional Networks for Large-Scale Image Recognition. 
    http://arxiv.org/abs/1409.1556
"""

from torch import nn, Tensor


class VGG(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        layers: list[int],
        use_batchnorm: bool = False,
    ) -> None:
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            VGGBlock(num_channels, 64, layers[0], use_batchnorm),
            VGGBlock(64, 128, layers[1], use_batchnorm),
            VGGBlock(128, 256, layers[2], use_batchnorm),
            VGGBlock(256, 512, layers[3], use_batchnorm),
            VGGBlock(512, 512, layers[4], use_batchnorm),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, images: Tensor):
        return self.model(images)


class VGGBlock(nn.Module):
    """VGG Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        use_batchnorm: bool = False,
    ) -> None:
        """Initialize a VGG Block.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            num_convs (int): number of convolution layers.
        """
        super(VGGBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        for _ in range(num_convs - 1):
            layers.append(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.block(x)


def vgg11(
    num_channels: int = 3, num_classes: int = 1000, use_batchnorm: bool = False
) -> VGG:
    return VGG(num_channels, num_classes, [1, 1, 2, 2, 2], use_batchnorm)


def vgg13(
    num_channels: int = 3, num_classes: int = 1000, use_batchnorm: bool = False
) -> VGG:
    return VGG(num_channels, num_classes, [2, 2, 2, 2, 2], use_batchnorm)


def vgg16(
    num_channels: int = 3, num_classes: int = 1000, use_batchnorm: bool = False
) -> VGG:
    return VGG(num_channels, num_classes, [2, 2, 3, 3, 3], use_batchnorm)


def vgg19(
    num_channels: int = 3, num_classes: int = 1000, use_batchnorm: bool = False
) -> VGG:
    return VGG(num_channels, num_classes, [2, 2, 4, 4, 4], use_batchnorm)
