import os
import argparse
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnext import resnext50, resnext101, resnext152
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models.mobilenetv1 import MobileNetV1
from models.mobilenetv2 import MobileNetV2
from models.vit import vit_extra_small, vit_small, vit_base, vit_large, vit_huge
from utils.dataset import DatasetWrapper
from utils.transform import dataset2transform

# Parse commanline arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_epochs",
    type=int,
    default=200,
    help="number of training epochs. defaults to 200",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.1,
    help="base learning rate for SGD optimizer. defaults to 0.1",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="momentum term for SGD optimizer. defaults to 0.9",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="weight decay coefficient. defaults to 0.0005",
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="mini-batch size. defaults to 128"
)
parser.add_argument(
    "--model", type=str, default="resnet50", help="model name. defaults to 'resnet50'"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="training dataset. defaults to 'cifar10'",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="runs",
    help="root directory for saving running log. defaults to 'runs'",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="checkpoints",
    help="root directory for saving model checkpoints. defaults to 'checkpoints'",
)
args = parser.parse_args()
print(args)

# Create folder if not exists
if not os.path.exists(os.path.join(args.ckpt_dir, args.dataset)):
    os.makedirs(os.path.join(args.ckpt_dir, args.dataset))


writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, args.model))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if args.dataset == "cifar10":
    # CIFAR-10 dataset
    num_classes = 10
    full_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
elif args.dataset == "cifar100":
    num_classes = 100
    full_dataset = datasets.CIFAR100(root="./data", train=True, download=True)

# Split dataset into training and validaion set(9 : 1)
train_dataset, val_dataset = random_split(full_dataset, [0.9, 0.1])

# Apply different data augmentation
train_dataset = DatasetWrapper(
    train_dataset, transform=dataset2transform[args.dataset]["train"]
)
val_dataset = DatasetWrapper(
    val_dataset, transform=dataset2transform[args.dataset]["val"]
)

# Data loader
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)


# ================ Model - START ================
if args.model == "resnet18":
    model = resnet18(num_classes=num_classes).to(device)
elif args.model == "resnet34":
    model = resnet34(num_classes=num_classes).to(device)
elif args.model == "resnet50":
    model = resnet50(num_classes=num_classes).to(device)
elif args.model == "resnet101":
    model = resnet101(num_classes=num_classes).to(device)
elif args.model == "resnet152":
    model = resnet152(num_classes=num_classes).to(device)
elif args.model == "resnext50":
    model = resnext50(num_classes=num_classes).to(device)
elif args.model == "resnext101":
    model = resnext101(num_classes=num_classes).to(device)
elif args.model == "resnext152":
    model = resnext152(num_classes=num_classes).to(device)
elif args.model == "mobilenetv1":
    model = MobileNetV1(num_classes=num_classes).to(device)
elif args.model == "mobilenetv2":
    model = MobileNetV2(num_classes=num_classes).to(device)
elif args.model == "vgg11":
    model = vgg11_bn(num_classes=num_classes).to(device)
elif args.model == "vgg13":
    model = vgg13_bn(num_classes=num_classes).to(device)
elif args.model == "vgg16":
    model = vgg16_bn(num_classes=num_classes).to(device)
elif args.model == "vgg19":
    model = vgg19_bn(num_classes=num_classes).to(device)
elif args.model == "vit_xs":
    model = vit_extra_small(image_size=32, num_channels=3).to(device)
elif args.model == "vit_s":
    model = vit_small(image_size=32, num_channels=3).to(device)
elif args.model == "vit_base":
    model = vit_base(image_size=32, num_channels=3).to(device)
elif args.model == "vit_large":
    model = vit_large(image_size=32, num_channels=3).to(device)
elif args.model == "vit_huge":
    model = vit_huge(image_size=32, num_channels=3).to(device)
else:
    raise RuntimeError(f"Unkown model: {args.model}")

# ================ Model - END ================


# ================ Training - START ================

# loss function
criterion = nn.CrossEntropyLoss().to(device)
# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)


def lr_rule(epoch: int) -> float:
    if epoch < 10:
        return (epoch + 1) / 10
    else:
        return 0.1 ** ((epoch + 1 - 10) // 60)


# scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_rule)


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
for epoch in range(args.num_epochs):
    # train
    train_loss, train_acc = train()
    print(
        f"Epoch: [{epoch + 1}]/[{args.num_epochs}], Training Loss: {train_loss:.4f}, Traing Accuracy: {100 * train_acc:.2f} %"
    )
    writer.add_scalar("training loss", train_loss, epoch + 1)
    writer.add_scalar("training accuracy", train_acc, epoch + 1)

    # validate
    val_loss, val_acc = val()
    print(
        f"Epoch: [{epoch + 1}]/[{args.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {100 * val_acc:.2f} %"
    )
    writer.add_scalar("validation loss", val_loss, epoch + 1)
    writer.add_scalar("validation accuracy", val_acc, epoch + 1)

    # Save the best model
    if val_acc > best_val_acc:
        torch.save(
            model.state_dict(),
            os.path.join(args.ckpt_dir, args.dataset, f"{args.model}_best.pth"),
        )
        best_val_acc = val_acc
    # Save latest model
    torch.save(
        model.state_dict(),
        os.path.join(args.ckpt_dir, args.dataset, f"{args.model}_latest.pth"),
    )
    scheduler.step()

# ================ Training - END ================
