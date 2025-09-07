# from .ResNet import *
# from .ResNets import *
# from .VGG import *
# from .VGG_LTH import *
from .ResNet import *
from .lsq_quant_resnet import *
from .mobilenetv2 import *

model_dict = {
    # "resnet18": resnet18,
    # "resnet50": resnet50,
    # "resnet20s": resnet20s,
    # "resnet44s": resnet44s,
    # "resnet56s": resnet56s,
    # "vgg16_bn": vgg16_bn,
    # "vgg16_bn_lth": vgg16_bn_lth,
    "resnet18": ResNet18,
    "Qresnet18": QResNet18,
    "mobilenetv2": Mobilenetv2,
    "QMobilenetv2": QMobilenetv2
}
