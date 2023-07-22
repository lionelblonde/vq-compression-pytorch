import torch.nn as nn

from helpers.model_util import ResnetToolkit


class ClassifierModelTenChan(nn.Module):

    def __init__(self, backbone_name, backbone_pretrained, fc_out_dim):
        super().__init__()

        self.resnet_league = ResnetToolkit.resnet_league(backbone_pretrained)

        if backbone_name not in self.resnet_league.keys():
            raise KeyError(f"invalid backbone name: {backbone_name}")

        self.backbone = self.resnet_league[backbone_name]
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, fc_out_dim)

        # change the number of input channels to be 10 instread of the default 3 for resnets
        in_chan = 10
        self.backbone.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)


    def forward(self, x):
        return self.backbone(x)

