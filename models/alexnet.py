"""
PyTorch implementation for AlexNet.

For more details, see:
[1] Alex Krizhevsky et.al https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""

from torch import nn
from torch import Tensor


class AlexNet(nn.Module):
    """AlexNet.

    Architecture: [conv - relu - max pool] x 2 - [conv - relu] x 3 - max pool - [affine - dropout] x 2 - affine - softmax
    """

    def __init__(self, num_channels: int, num_classes: int = 10) -> None:
        """Initialize AlexNet.

        Args:
            num_channels(int): number of channels of input images.
            num_classes(int): number of classes.
        """
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),  # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),  # 27x27
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 13x13
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 13x13
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 13x13
            nn.MaxPool2d(kernel_size=3, stride=2),  # 6x6
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)
