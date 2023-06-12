from copy import deepcopy

import torch.nn as nn
import torchvision


def init(weight_scale=1., constant_bias=0.):
    """Perform orthogonal initialization"""

    def _init(m):

        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=weight_scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, nn.BatchNorm2d) or
              isinstance(m, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


def conv_to_fc(x, in_width, in_height):
    assert (isinstance(x, nn.Sequential) or
           isinstance(x, nn.Module))
    specs = [[list(i) for i in [mod.kernel_size,
              mod.stride,
              mod.padding]]
             for mod in x.modules()
             if isinstance(mod, nn.Conv2d)]
    acc = [deepcopy(in_width),
           deepcopy(in_height)]
    for e in specs:
        for i, (k, s, p) in enumerate(zip(*e)):
            acc[i] = ((acc[i] - k + (2 * p)) // s) + 1
    return acc[0] * acc[1]


class ResnetToolkit(object):

    @staticmethod
    def resnet_league(backbone_pretrained):
        return {
            "resnet18": torchvision.models.resnet18({
                "weights": (torchvision.models.ResNet18_Weights.DEFAULT
                            if backbone_pretrained
                            else None),
            }),
            "resnet50": torchvision.models.resnet50({
                "weights": (torchvision.models.ResNet50_Weights.DEFAULT
                            if backbone_pretrained
                            else None),
            }),
            "resnet101": torchvision.models.resnet101({
                "weights": (torchvision.models.ResNet101_Weights.DEFAULT
                            if backbone_pretrained
                            else None),
            }),
            "resnet152": torchvision.models.resnet152({
                "weights": (torchvision.models.ResNet152_Weights.DEFAULT
                            if backbone_pretrained
                            else None),
            }),
        }
