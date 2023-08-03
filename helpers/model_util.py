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
    skip_list=(),
):
    decay = []
    no_decay = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (len(p.size()) == 1 or n in skip_list):
            # covers batchnorm params and biases
            # or obviously is named by name in skip list
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': no_decay,
         'weight_decay': 0.,
         'lars': False},  # we never layer-adapt when not ok for wd
        {'params': decay,
         'weight_decay': weight_decay,
         'lars': True},  # we "can" layer-adapt when ok for wd
    ]
