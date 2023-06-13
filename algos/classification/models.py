from collections import OrderedDict
import math

import torch
import torch.nn as nn

from helpers.model_util import ResnetToolkit, init


# ---------- classifier

class ClassifierModel(nn.Module):

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


class PaperClassifierModel(nn.Module):

    def __init__(self, fc_hid_dim, fc_out_dim):
        super().__init__()

        in_chan = 10

        # Assemble the convolutional stack
        self.conv2d_stack = nn.Sequential(OrderedDict([
            ('conv2d_block_1', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(in_chan, 32, kernel_size=5, stride=1, padding=1)),
                ('bn', nn.BatchNorm2d(32)),
                ('nl', nn.ReLU()),
                ('mp', nn.MaxPool2d(kernel_size=2)),
                # ('do', nn.Dropout(p=0.25)),
            ]))),
            ('conv2d_block_2', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)),
                ('bn', nn.BatchNorm2d(64)),
                ('nl', nn.ReLU()),
                ('mp', nn.MaxPool2d(kernel_size=2)),
                # ('do', nn.Dropout(p=0.25)),
            ]))),
            ('conv2d_block_3', nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
                ('bn', nn.BatchNorm2d(64)),
                ('nl', nn.ReLU()),
                # ('do', nn.Dropout(p=0.25)),
            ]))),
        ]))
        # Assemble the fully-connected stack
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(50176, fc_hid_dim)),  # HAXX: found the number in error message
                ('nl', nn.ReLU()),
                # ('do', nn.Dropout(p=0.5)),
            ]))),
        ]))
        # Assemble the output classification head
        self.c_head = nn.Linear(fc_hid_dim, fc_out_dim)
        # Perform initialization
        self.conv2d_stack.apply(init(weight_scale=math.sqrt(2)))
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.c_head.apply(init(weight_scale=0.01))

    def forward(self, x):
        # Stack the convolutional layers
        x = self.conv2d_stack(x)
        # Flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        # Stack the fully-connected layers
        x = self.fc_stack(x)
        # Return the head
        return self.c_head(x)
