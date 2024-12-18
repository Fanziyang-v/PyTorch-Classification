import os
import argparse
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnext import resnext50, resnext101, resnext152
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models.mobilenetv1 import MobileNetV1
from models.mobilenetv2 import MobileNetV2
from models.vit import vit_extra_small, vit_small, vit_base, vit_large, vit_huge
from utils.transform import dataset2transform

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parse commanline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model type")
parser.add_argument("--dataset", type=str, help="dataset name(cifar10 | cifar100)")
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="checkpoints",
    help="root directory for saving model checkpoints",
)
args = parser.parse_args()
print(args)

# ================ Dataset - START ================
if args.dataset not in ("cifar10", "cifar100"):
    raise RuntimeError(f"Unkown dataset: {args.dataset}")

num_classes = 10 if args.dataset == "cifar10" else 100
if args.dataset == "cifar10":
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=dataset2transform[args.dataset]["test"]
    )
else:
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, transform=dataset2transform[args.dataset]["test"]
    )
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=128, shuffle=False, num_workers=4
)
# ================ Dataset - END ================
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
# ================ Model - START ================
model.load_state_dict(
    torch.load(os.path.join(args.ckpt_dir, args.dataset, f"{args.model}_best.pth"))
)

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
