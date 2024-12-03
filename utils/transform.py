from torchvision import transforms

dataset2transform: dict[str, dict[str, transforms.Compose]] = {
    "cifar10": {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
                ),
            ]
        ),
    },
    "cifar100": {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
                ),
            ]
        ),
    },
}
