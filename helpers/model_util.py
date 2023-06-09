import torchvision


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
