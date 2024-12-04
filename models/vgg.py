"""
PyTorch implementation for VGGNet.

For more details, see:
[1] Karen Simonyan et al. 
    Very Deep Convolutional Networks for Large-Scale Image Recognition. 
    http://arxiv.org/abs/1409.1556
"""

from torch import nn, Tensor


class VGG(nn.Module):
    """VGG Net.

    Unlike the original VGG Net, a global average pooling is applied
    after the fifth VGG block and before the first fully connected layer,
    resulting in the spatial dimension of the output feature map is 1x1

    Also, we don't use max pooling layer in the first two VGG Blocks.
    """

    def __init__(
        self,
        num_classes: int,
        layers: list[int],
        use_batchnorm: bool = False,
    ) -> None:
        """Initialize a VGG Net with optional batch normalization layer.

        Args:
            num_classes (int): number of classes of input images.
            layers (list[int]): number of conv layers in each VGG block.
            use_batchnorm (bool, optional): batchnorm will be applided if use_batchnorm is True. Defaults to False.
        """
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            VGGBlock(3, 64, layers[0], use_batchnorm),
            VGGBlock(64, 128, layers[1], use_batchnorm),
            VGGBlock(128, 256, layers[2], use_batchnorm, use_pool=True),
            VGGBlock(256, 512, layers[3], use_batchnorm, use_pool=True),
            VGGBlock(512, 512, layers[4], use_batchnorm, use_pool=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def forward(self, images: Tensor):
        return self.model(images)

    def _init_weights(self, mean: float = 0, std: float = 0.1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=mean, std=std)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=mean, std=std)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class VGGBlock(nn.Module):
    """VGG Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        use_batchnorm: bool = False,
        use_pool: bool = False
    ) -> None:
        """Initialize a VGG Block.

        Args:
            in_channels (int): number of channels of input feature map.
            out_channels (int): number of channels of output feature map.
            num_convs (int): number of convolution layers.
        """
        super(VGGBlock, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_convs - 1):
            layers.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.block(x)


def vgg11(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [1, 1, 2, 2, 2])


def vgg13(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [2, 2, 2, 2, 2])


def vgg16(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [2, 2, 3, 3, 3])


def vgg19(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [2, 2, 4, 4, 4])


def vgg11_bn(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [1, 1, 2, 2, 2], use_batchnorm=True)


def vgg13_bn(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [2, 2, 2, 2, 2], use_batchnorm=True)


def vgg16_bn(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [2, 2, 3, 3, 3], use_batchnorm=True)


def vgg19_bn(num_classes: int = 10) -> VGG:
    return VGG(num_classes, [2, 2, 4, 4, 4], use_batchnorm=True)
