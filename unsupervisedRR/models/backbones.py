from torch import nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3, resnet18


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_c, out_c, batchnorm=True, activation=True, k=3):
        super().__init__()
        if k == 3:
            self.conv = conv3x3(in_c, out_c)
        elif k == 1:
            self.conv = conv1x1(in_c, out_c)
        else:
            raise ValueError()

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_c)
        else:
            self.bn = nn.Identity()

        if activation:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_c, out_c, upsampling_method):
        super().__init__()
        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1),
            )

    def forward(self, x):
        return self.upsample(x)


class ResNetEncoder(nn.Module):
    def __init__(self, chan_in=3, chan_out=64, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.inconv = ConvBlock(chan_in, 64, k=3)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer1
        self.outconv = ConvBlock(64, chan_out, k=1, activation=False)

    def forward(self, x):
        # first apply the resnet blocks up to layer 2
        x = self.inconv(x)
        x = self.layer1(x)  # -> 64 x H/2 x W/2
        x = self.layer2(x)  # -> 64 x H/2 x W/2
        x = self.outconv(x)

        return x


class ResNetDecoder(nn.Module):
    def __init__(self, chan_in, chan_out, non_linearity, pretrained=False):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        resnet.inplanes = chan_in
        self.layer1 = resnet._make_layer(BasicBlock, 64, 2)
        resnet.inplanes = 64
        self.layer2 = resnet._make_layer(BasicBlock, 64, 2)

        self.upconv1 = UpConv(64, 64, "bilinear")
        self.outconv = ConvBlock(64, chan_out, batchnorm=False, activation=False)

        if non_linearity is None:
            self.non_linearity = nn.Identity()
        else:
            self.non_linearity = non_linearity

        # Initialize all the new layers
        self.resnet_init()

    def resnet_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.layer1(x)  # -> 128 x H/4 x W/4
        x = self.layer2(x)  # -> 64 x H/2 x W/2
        x = self.outconv(x)  # -> C_out x H x W
        x = self.non_linearity(x)
        return x
