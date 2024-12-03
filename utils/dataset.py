from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    """Dataset wrapper for further data augmentation"""

    def __init__(self, dataset: Dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
