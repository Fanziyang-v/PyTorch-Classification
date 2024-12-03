import os
import argparse
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import datasets, transforms
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parse commanline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model type")
parser.add_argument("--dataset", type=str, help="dataset name(cifar10 | cifar100)")
parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="root directory for saving model checkpoints")
args = parser.parse_args()
print(args)

# ================ Dataset - START ================
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2467, 0.2432, 0.2612)
        ),
    ]
)
if args.dataset == "cifar10":
    num_classes = 10
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform)
elif args.dataset == "cifar100":
    num_classes = 100
    test_dataset = datasets.CIFAR100(root="./data", train=False, transform=transform)
else:
    raise RuntimeError(f"Unkown dataset: {args.dataset}")
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4
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
else:
    raise RuntimeError(f"Unkown model: {args.model}")
# ================ Model - START ================
model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.dataset, f"{args.model}_best.pth")))

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
