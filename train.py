import os
import argparse
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# Hyper parameters
num_epochs = 200
batch_size = 256
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
dataset = "cifar10"
model_name = "resnet50"
log_dir = "runs"
ckpt_dir = "checkpoints"

if not os.path.exists(os.path.join(ckpt_dir, dataset)):
    os.makedirs(os.path.join(ckpt_dir, dataset))


writer = SummaryWriter(log_dir)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class DatasetWrapper(Dataset):
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


# Image preprocessing
train_transfrom = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
        ),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
        ),
    ]
)

# CIFAR-10 dataset
full_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
# Split dataset into training and validaion set(9 : 1)
train_dataset, val_dataset = random_split(full_dataset, [0.9, 0.1])

# Apply different data augmentation
train_dataset = DatasetWrapper(train_dataset, transform=train_transfrom)
val_dataset = DatasetWrapper(val_dataset, transform=val_transform)

# Data loader
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# model
model = resnet50(num_classes=10).to(device)
# loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1)


def train():
    model.train()
    train_loss = train_acc = 0
    for images, labels in train_dataloader:
        images: Tensor = images.to(device)
        labels: Tensor = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss: Tensor = criterion(outputs, labels)
        pred = torch.argmax(outputs, dim=1)
        acc = torch.mean((pred == labels).float())

        # Backprop and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record training loss and accuracy
        train_loss += loss.item()
        train_acc += acc.item()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    return train_loss, train_acc


def val() -> tuple[float, float]:
    model.eval()
    val_loss = val_acc = 0
    for images, labels in val_dataloader:
        images: Tensor = images.to(device)
        labels: Tensor = labels.to(device)
        outputs = model(images)
        loss: Tensor = criterion(outputs, labels)
        pred = torch.argmax(outputs, dim=1)
        acc = torch.mean((pred == labels).float())

        val_acc += acc.item()
        val_loss += loss.item()
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)
    return val_loss, val_acc


# Start training
best_val_acc = 0
for epoch in range(num_epochs):
    # train
    train_loss, train_acc = train()
    print(
        f"Epoch: [{epoch + 1}]/[{num_epochs}], Training Loss: {train_loss:.4f}, Traing Accuracy: {100 * train_acc:.2f} %"
    )
    writer.add_scalar("training loss", train_loss, epoch + 1)
    writer.add_scalar("training accuracy", train_acc, epoch + 1)

    # validate
    val_loss, val_acc = val()
    print(
        f"Epoch: [{epoch + 1}]/[{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {100 * val_acc:.2f} %"
    )
    writer.add_scalar("validation loss", val_loss, epoch + 1)
    writer.add_scalar("validation accuracy", val_acc, epoch + 1)

    # Save the best model
    if val_acc > best_val_acc:
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, dataset, f"{model_name}_best.pth"),
        )
        best_val_acc = val_acc
    # Save latest model
    torch.save(
        model.state_dict(), os.path.join(ckpt_dir, dataset, f"{model_name}_latest.pth")
    )
    scheduler.step(metrics=val_loss)
