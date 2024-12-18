from models.densenet import densenet121, densenet169, densenet201, densenet264
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnext import resnext50, resnext101, resnext152
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models.mobilenetv1 import MobileNetV1
from models.mobilenetv2 import MobileNetV2
from models.xception import Xception
from models.googlenet import GoogLeNet


def get_model(model_name: str):
    """Construct model by name."""
    match model_name:
        case "densenet121":
            model = densenet121(num_classes=10)
        case "densenet169":
            model = densenet169(num_classes=10)
        case "densenet201":
            model = densenet201(num_classes=10)
        case "densenet264":
            model = densenet264(num_classes=10)
        case "resnet18":
            model = resnet18(num_classes=10)
        case "resnet34":
            model = resnet34(num_classes=10)
        case "resnet50":
            model = resnet50(num_classes=10)
        case "resnet101":
            model = resnet101(num_classes=10)
        case "resnet152":
            model = resnet152(num_classes=10)
        case "resnext50":
            model = resnext50(num_classes=10)
        case "resnext101":
            model = resnext101(num_classes=10)
        case "resnext152":
            model = resnext152(num_classes=10)
        case "vgg11":
            model = vgg11_bn(num_classes=10)
        case "vgg13":
            model = vgg13_bn(num_classes=10)
        case "vgg16":
            model = vgg16_bn(num_classes=10)
        case "vgg19":
            model = vgg19_bn(num_classes=10)
        case "mobilenetv1":
            model = MobileNetV1(num_classes=10)
        case "mobilenetv2":
            model = MobileNetV2(num_classes=10)
        case "xception":
            model = Xception(num_channels=3, num_classes=10)
        case "googlenet":
            model = GoogLeNet(num_classes=10)
        case _:
            raise RuntimeError(f"Unkown model: {model_name}")
    return model
