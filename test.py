import os
import argparse
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
        ),
    ]
)

test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4
)

model = resnet50(num_classes=10).to(device)

num_images = num_correct_top1 = num_correct_top5 = 0
for images, labels in test_dataloader:
    images: Tensor = images.to(device)
    labels: Tensor = labels.to(device)
    outputs = model(images)
    pred = torch.argsort(outputs, dim=1, descending=True)
    # Top-1 accuracy
    num_correct_top1 += torch.sum((pred[:, 0] == labels).float())
    # Top-5 accuracy
    labels = labels.view(-1, 1)
    num_correct_top5 += torch.sum((pred[:, :5] == labels).float())
    num_images += len(images)

acc_top1 = num_correct_top1 / num_images
acc_top5 = num_correct_top5 / num_images

print(f"Top-1 Accuracy: {acc_top1}, Top-5 Accuracy: {acc_top5}")
