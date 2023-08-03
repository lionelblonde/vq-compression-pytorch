import torch.nn as nn
import torchvision


class ResnetToolkit(object):

    @staticmethod
    def resnet_league(backbone_pretrained):
        return {
            "resnet18": torchvision.models.resnet18(
                weights=(torchvision.models.ResNet18_Weights.DEFAULT
                         if backbone_pretrained
                         else None),
            ),
            "resnet50": torchvision.models.resnet50(
                weights=(torchvision.models.ResNet50_Weights.DEFAULT
                         if backbone_pretrained
                         else None),
            ),
            "resnet101": torchvision.models.resnet101(
                weights=(torchvision.models.ResNet101_Weights.DEFAULT
                         if backbone_pretrained
                         else None),
            ),
            "resnet152": torchvision.models.resnet152(
                weights=(torchvision.models.ResNet152_Weights.DEFAULT
                         if backbone_pretrained
                         else None),
            ),
        }


def add_weight_decay(
    # not only does it prep for wd, but also for LARS opt
    model,
    weight_decay,
    skip_list=(nn.InstanceNorm1d, nn.BatchNorm1d,
               nn.InstanceNorm2d, nn.BatchNorm2d),
    using_lars=False,
):
    decay = []
    no_decay = []
    for module in model.modules():
        if not using_lars:
            params = [p for p in module.parameters() if p.requires_grad]
        else:
            # if using LARS, also remove the biases
            params = [p for n, p in module.named_parameters()
                      if (p.requires_grad and "bias" not in n)]
        if isinstance(module, skip_list):
            no_decay.extend(params)
        else:
            decay.extend(params)
    return [
        {'params': no_decay,
         'weight_decay': 0.,
         'lars': False},  # we never layer-adapt when not ok for wd
        {'params': decay,
         'weight_decay': weight_decay,
         'lars': True},  # we "can" layer-adapt when ok for wd
    ]
