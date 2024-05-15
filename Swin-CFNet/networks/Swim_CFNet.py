import torch
import torch.nn as nn
import numpy as np
# from resnet import resnet34
from networks.Encoder import Swim_Transformer
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()

        self.conv1 = Conv2dReLU(
            in_channels//2 + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = Conv2dReLU(
            in_channels//2,
            in_channels//2,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = Conv2dReLU(
            64,
            64,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conT=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)

        self.conT1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
    def forward(self, x, skip=None):
        x = self.conT(x)
        # print(x.shape)


        if skip is not None:
            #skip = self.cbam(skip)  #sptial attention

            x = torch.cat([x, skip], dim=1)

            x = self.conv1(x)

            x = self.conv2(x)
        else:
            # print("aaa")
            x=self.conv3(x)
            x = self.conv4(x)
            x = self.conT1(x)

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 1024
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        #print('-: ',in_channels)
        #in_channels=[512,512,256,128,64]
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            #print('self.config.n_skip',self.config.n_skip) 3
            skip_channels = self.config.skip_channels
            #print('self.config.n_skip', self.config.skip_channels)[512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip 4
                skip_channels[3-i]=0  #skip_channels[3]=0  i=0 3#[512,256,128,64,16]
        else:
            skip_channels=[0,0,0,0]

        #print(in_channels,out_channels, skip_channels) #[512, 256, 128, 64] (256, 128, 64, 16) [512, 256, 64, 0]
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.conv_more = Conv2dReLU(1024, 1024, kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, x, features=None):
        #B, n_patch, hidden = hidden_states.size()  # [12, 256, 768] reshape from (B, n_patch, hidden) to (B, h, w, hidden)

        x = self.conv_more(x)
        # print(x.shape)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None #config.n_skip = 3
                #print('ss:',skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
class Swim_CFNet(nn.Module):
    def __init__(self, config, num_classes=5, zero_head=False):
        super(Swim_CFNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.transformer = Swim_Transformer(config,block_units=config.resnet.num_layers,
                                            width_factor=config.resnet.width_factor)

        self.decoder = DecoderCup(config)
        # self.resnet=Resnet34(in_channel=4,out_channle=num_classes,pretrain=False)
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        trans_x=x
        feature,features= self.transformer(trans_x)
        # print(len(features))
        # print(feature.shape)
        x = self.decoder(feature, features)
        # print(x.shape)
        logits = self.segmentation_head(x)
        return {"out": logits}
