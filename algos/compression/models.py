import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, num_conv2d, relu, **kwargs):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('conv2d_0', nn.Conv2d(**kwargs))
        for i, _ in enumerate(range(num_conv2d - 1), start=1):
            self.layers.add_module(f"bn2d_{i}", nn.BatchNorm2d(kwargs['out_channels']))
            if relu:
                self.layers.add_module(f"relu_{i}", nn.ReLU())
            self.layers.add_module(f"conv2d_{i}", nn.Conv2d(**kwargs))
        self.layers.add_module('bn2d_f', nn.BatchNorm2d(kwargs['out_channels']))  # final

    def forward(self, x):
        return self.layers(x) + x  # residual connection


class EncoderModel(nn.Module):

    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        n = hps.ae_hidden  # output channels
        k = hps.ae_kernel  # kernel sizes
        p = ((k + 1) // 2) - 1  # padding

        # Create the two first layers, which can be/are downsampling ones

        self.layer_zero = nn.Sequential()
        kwargs = dict(
            in_channels=hps.in_channels,
            out_channels=n // 2,
            bias=False,
        )
        if hps.dsf in [8, 4, 2]:
            self.layer_zero.add_module(
                'conv2d_0',
                nn.Conv2d(
                    **kwargs,
                    kernel_size=k,
                    stride=2,
                    padding=p,
                ),
            )
        else:  # dsf is 1: no downsampling at all
            self.layer_zero.add_module(
                'conv2d_0',
                nn.Conv2d(
                    **kwargs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.layer_zero.add_module('bn2d_0', nn.BatchNorm2d(n // 2))
        self.layer_zero.add_module('relu_0', nn.ReLU())

        self.layer_one = nn.Sequential()
        kwargs = dict(
            in_channels=n // 2,
            out_channels=n,
            bias=False,
        )
        if hps.dsf == 8:
            self.layer_one.add_module(
                'conv2d_1',
                nn.Conv2d(
                    **kwargs,
                    kernel_size=k,
                    stride=2,
                    padding=p,
                ),
            )
        else:
            self.layer_one.add_module(
                'conv2d_1',
                nn.Conv2d(
                    **kwargs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.layer_one.add_module('bn2d_1', nn.BatchNorm2d(n))
        self.layer_one.add_module('relu_1', nn.ReLU())

        # Create the Resnet blocks

        self.resblocks = nn.Sequential()
        kwargs = dict(
            num_conv2d=2,
            in_channels=n,
            out_channels=n,
            kernel_size=3,
            padding=1,
        )
        for i in range(hps.ae_resblocks):
            layers = nn.Sequential()
            for j in range(3):
                layers.add_module(
                    f"resblock_{i}_{j}",
                    ResBlock(
                        **kwargs,
                        relu=True,
                    ),
                )
            self.resblocks.add_module(f"resblock_{i}", layers)
        self.resblocks.add_module(
            "resblock_f",
            ResBlock(
                **kwargs,
                relu=False,
            ),
        )

        # Create the final layer, with special treating if high downsampling is chosen

        self.layer_final = nn.Sequential()
        kwargs = dict(
            in_channels=n,
            out_channels=hps.z_channels,
            bias=False,
        )
        if hps.dsf in [8, 4]:
            self.layer_final.add_module(
                'conv2d_1',
                nn.Conv2d(
                    **kwargs,
                    kernel_size=k,
                    stride=2,
                    padding=p,
                ),
            )
        else:
            self.layer_final.add_module(
                'conv2d_1',
                nn.Conv2d(
                    **kwargs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.layer_final.add_module('bn2d_f', nn.BatchNorm2d(hps.z_channels))

    def forward(self, x):
        x = self.layer_zero(x)
        x = self.layer_one(x)
        x_skip = x
        for b in range(self.hps.ae_resblocks):
            x = self.resblocks[b](x) + x
        x = self.resblocks[b + 1](x) + x_skip
        x = self.layer_final(x)
        return x


class DecoderModel(nn.Module):

    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        n = hps.ae_hidden  # output channels
        k = hps.ae_kernel  # kernel sizes
        p = ((k + 1) // 2) - 1  # padding

        # Create the first layer, which can be/are downsampling ones

        self.layer_zero = nn.Sequential()
        kwargs = dict(
            in_channels=hps.z_channels,
            out_channels=n,
            bias=False,
        )
        if hps.dsf in [8, 4]:
            self.layer_zero.add_module(
                'convtransp2d_0',
                nn.ConvTranspose2d(
                    **kwargs,
                    kernel_size=k,
                    stride=2,
                    padding=p,
                ),
            )
        else:  # dsf is 1: no downsampling at all
            self.layer_zero.add_module(
                'convtransp2d_0',
                nn.ConvTranspose2d(
                    **kwargs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.layer_zero.add_module('bn2d_0', nn.BatchNorm2d(n))
        self.layer_zero.add_module('relu_0', nn.ReLU())

        # Create the Resnet blocks

        self.resblocks = nn.Sequential()
        kwargs = dict(
            num_conv2d=2,
            in_channels=n,
            out_channels=n,
            kernel_size=3,
            padding=1,
        )
        for i in range(hps.ae_resblocks):
            layers = nn.Sequential()
            for j in range(3):
                layers.add_module(
                    f"resblock_{i}_{j}",
                    ResBlock(
                        **kwargs,
                        relu=True,
                    ),
                )
            self.resblocks.add_module(f"resblock_{i}", layers)
        self.resblocks.add_module(
            "resblock_f",
            ResBlock(
                **kwargs,
                relu=False,
            ),
        )

        # Create the two last layers, which can be/are downsampling ones

        self.layer_penultimate = nn.Sequential()
        kwargs = dict(
            in_channels=n,
            out_channels=n // 2,
            bias=False,
        )
        if hps.dsf == 8:
            self.layer_penultimate.add_module(
                'convtransp2d_p',
                nn.ConvTranspose2d(
                    **kwargs,
                    kernel_size=k,
                    stride=2,
                    padding=p,
                ),
            )
        else:
            self.layer_penultimate.add_module(
                'convtransp2d_p',
                nn.ConvTranspose2d(
                    **kwargs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.layer_penultimate.add_module('bn2d_p', nn.BatchNorm2d(n // 2))
        self.layer_penultimate.add_module('relu_p', nn.ReLU())

        self.layer_final = nn.Sequential()
        kwargs = dict(
            in_channels=n // 2,
            out_channels=hps.in_channels,
            bias=False,
        )
        if hps.dsf in [8, 4, 2]:
            self.layer_final.add_module(
                'convtransp2d_f',
                nn.ConvTranspose2d(
                    **kwargs,
                    kernel_size=k,
                    stride=2,
                    padding=p,
                ),
            )
        else:  # dsf is 1: no downsampling at all
            self.layer_final.add_module(
                'convtransp2d_f',
                nn.ConvTranspose2d(
                    **kwargs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.layer_final.add_module('bn2d_f', nn.BatchNorm2d(hps.in_channels))

    def forward(self, x):
        x = self.layer_zero(x)
        x_skip = x
        for b in range(self.hps.ae_resblocks):
            x = self.resblocks[b](x) + x
        x = self.resblocks[b + 1](x) + x_skip
        x = self.layer_penultimate(x)
        x = self.layer_final(x)
        return x
