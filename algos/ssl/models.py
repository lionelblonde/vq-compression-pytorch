import torch.nn as nn

from helpers.model_util import ResnetToolkit


class SimCLRModel(nn.Module):

    def __init__(self, backbone_name, backbone_pretrained, fc_hid_dim, fc_out_dim):
        super().__init__()

        self.resnet_league = ResnetToolkit.resnet_league(backbone_pretrained)
        if backbone_name not in self.resnet_league.keys():
            raise KeyError(f"invalid backbone name: {backbone_name}")
        self.backbone = self.resnet_league[backbone_name]
        self.tv_backbone_inner_fc_dim = self.backbone.fc.in_features
        # this value was self-ed because used outside to create the linear probe

        # change the number of input channels to be 10 instread of the default 3 for resnets
        in_chan = 10
        self.backbone.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(self.tv_backbone_inner_fc_dim, fc_hid_dim, bias=False),
            nn.BatchNorm1d(fc_hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hid_dim, fc_out_dim, bias=True),
        )

    def forward(self, x_i, x_j):
        z_i = self.head(self.backbone(x_i))
        z_j = self.head(self.backbone(x_j))
        return z_i, z_j

    def mono_forward(self, x):
        h = self.backbone(x)
        z = self.head(h)
        return z

