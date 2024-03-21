
import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

class mix_channle_fuse2(nn.Module):
    def __init__(self,
                 image_size: int = 128,
                 input_channle: int = 16,
                 deeps: Sequence[int] = [1, 2, 4, 8, 16],
                 group: Sequence[int] = [3,5,7,11],
                 up_size: int = 3,
                 ):
        super(mix_channle_fuse2, self).__init__()

        self.image_size = image_size
        self.input_channle = input_channle
        self.deeps = deeps
        self.num_deeps = len(self.deeps)
        self.num_group = len(group)
        self.num_patch = self.input_channle // self.num_group
        self.fuse_size = self.image_size // 2
        self.fuse_shape = [self.fuse_size, self.fuse_size, self.fuse_size]
        self.split_group = nn.ModuleList()
        self.fuse2 = nn.ModuleList()
        self.fuse3 = nn.ModuleList()
        self.mid_channle = 32

        for i in range(self.num_deeps):
            split_group = Mix_channle2(input_size=self.image_size // (2 ** i),
                                       input_channle=self.deeps[i] * self.input_channle,
                                       out_feature_size=self.mid_channle,
                                       fuse_feature_size=self.fuse_size,
                                       nth_layer=i)
            self.split_group.append(split_group)


            if i == 0:
                fuse2 = nn.Sequential(UpConv(input_channle=self.mid_channle*self.num_deeps*self.num_group,
                                                    out_feature_size=(self.deeps[i]+1) * self.input_channle,
                                                    kernel_size=up_size,
                                                    stride=2),
                                    UpConv(input_channle=(self.deeps[i]+1) * self.input_channle,
                                                    out_feature_size=self.deeps[i] * self.input_channle,
                                                    kernel_size=up_size,
                                                    stride=2),)
            if i == 1:
                fuse2 = nn.Sequential(UpConv(input_channle=self.mid_channle*self.num_deeps*self.num_group,
                                                    out_feature_size=self.deeps[i] * self.input_channle,
                                                    kernel_size=up_size,
                                                    stride=2),)
            if i == 2:
                fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle*self.num_deeps*self.num_group,
                                                    out_feature_size=self.deeps[i] * self.input_channle,
                                                    kernel_size=up_size,
                                                    stride=1), )
            if i == 3:
                fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle*self.num_deeps*self.num_group,
                                                    out_feature_size=self.deeps[i] * self.input_channle,
                                                    kernel_size=up_size,
                                                    stride=2),)
            if i == 4:
                fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle*self.num_deeps*self.num_group,
                                                    out_feature_size=(self.deeps[i]-1) * self.input_channle,
                                                    kernel_size=up_size,
                                                    stride=2),
                                           DownConv(input_channle=(self.deeps[i]-1) * self.input_channle,
                                                  out_feature_size=self.deeps[i] * self.input_channle,
                                                  kernel_size=up_size,
                                                  stride=2), )

            self.fuse2.append(fuse2)
            fuse3 = nn.Sequential(nn.Conv3d(self.deeps[i] * self.input_channle*2,
                                            self.deeps[i] * self.input_channle,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0),
                                  nn.GroupNorm(self.num_group,self.deeps[i] * self.input_channle),
                                  nn.GELU())
            self.fuse3.append(fuse3)
        self.pool = nn.AdaptiveAvgPool3d(self.fuse_shape)
        self.fuse = nn.Sequential(
            nn.Conv3d(self.num_group * self.num_deeps, self.num_group * self.num_deeps * 4, 1, 1, 0),
            nn.GroupNorm(self.num_group, self.num_group * self.num_deeps * 4),
            nn.GELU(),
            nn.Conv3d(self.num_group * self.num_deeps * 4, self.num_group * self.num_deeps, 1, 1, 0),
            nn.GroupNorm(self.num_group, self.num_group * self.num_deeps),
            nn.GELU(),
            )

    def forward(self, x):
        x1 = x
        for i in range(self.num_deeps):
            # print(i,)
            B, C, _, _, _ = x[i].shape
            x_abcd = self.split_group[i](x[i])
            B, channle_group, num_group, D, H, W = x_abcd.shape

            if i == 0:
                x_all = x_abcd
            else:
                x_all = torch.cat([x_all, x_abcd], dim=2)

        for i in range(self.num_patch):
            x_all[:, i, :] = self.fuse(x_all[:, i, :].clone())
        x_all = x_all.reshape(B, -1,D, H, W)

        for i in range(self.num_deeps):
            x_all_r = self.fuse2[i](x_all)
            y = x1[i]
            x1[i] =self.fuse3[i](torch.cat([x_all_r , y],dim=1))  # res

        return x


class DownConv(nn.Module):
    def __init__(self,
                 input_channle: int = 24,
                 out_feature_size: int = 24,
                 kernel_size: int = 7,
                 stride: int = 2,
                 group:int = 4):
        super(DownConv, self).__init__()

        self.conv1 = nn.Conv3d(input_channle,
                               input_channle,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=input_channle)
        self.conv2 = nn.Conv3d(input_channle, out_feature_size,
                               kernel_size=1,
                               stride=1,
                               padding=0, )
        self.norm = nn.GroupNorm(group, input_channle)
        self.act = nn.GELU()

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        return x1


class UpConv(nn.Module):
    def __init__(self,
                 input_channle: int = 24,
                 out_feature_size: int = 24,
                 kernel_size: int = 7,
                 stride: int = 2,
                 group: int =4):
        super(UpConv, self).__init__()

        self.conv1 = nn.ConvTranspose3d(input_channle,
                               input_channle,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=input_channle)
        self.conv2 = nn.Conv3d(input_channle, out_feature_size,
                               kernel_size=1,
                               stride=1,
                               padding=0, )
        self.norm = nn.GroupNorm(group, input_channle)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))
        return x1

def choic_layer(input_channle,output_channle,kernel_size,input_size,mid_size):
    layer = nn.Sequential()
    if input_size>mid_size:
        nth = input_size//mid_size
        for i in range(nth):
            conv = DownConv(input_channle=input_channle,
                            out_feature_size=output_channle,
                            kernel_size=kernel_size,
                            stride=2)

class Mix_channle2(nn.Module):
    def __init__(self,
                 input_size: int = 128,
                 input_channle: int = 24,
                 out_feature_size: int = 24,
                 fuse_feature_size: int = 64,
                 group: Sequence[int] = [1, 3, 5, 7],
                 image_size: int = 128,
                 nth_layer: int = 0):
        super(Mix_channle2, self).__init__()
        self.input_size = int(input_size)
        self.group = group
        self.num_group = len(group)  # 4
        self.channle_group = input_channle // self.num_group
        self.out_feature_size = out_feature_size
        self.fuse_feature_size = fuse_feature_size
        self.image_size = int(image_size)
        self.conv_group = nn.ModuleList()
        self.nth_layer = nth_layer
        self.nth = self.nth_layer_number()
        print(self.nth,self.input_size,self.fuse_feature_size,self.input_size//self.fuse_feature_size)
        for i in range(self.num_group):
            sub_conv_group = nn.ModuleList()
            if self.input_size>self.fuse_feature_size:
                nth = self.input_size//self.fuse_feature_size

                for j in range(nth+1):
                    conv_group = DownConv(input_channle=self.channle_group,
                                        out_feature_size=self.channle_group,
                                        kernel_size=self.group[i],
                                        stride=2)
                    sub_conv_group.append(conv_group)

            if self.nth_layer == 0:
                conv_group = nn.Sequential(DownConv(input_channle=self.channle_group,
                                                    out_feature_size=self.channle_group,
                                                    kernel_size=self.group[i],
                                                    stride=2),
                                           DownConv(input_channle=self.channle_group,
                                                    out_feature_size=self.out_feature_size ,
                                                    kernel_size=self.group[i],
                                                    stride=2),)
            if self.nth_layer == 1:
                conv_group = nn.Sequential(DownConv(input_channle=self.channle_group,
                                                    out_feature_size=self.out_feature_size,
                                                    kernel_size=self.group[i],
                                                    stride=2),)
            if self.nth_layer == 2:
                conv_group = nn.Sequential(DownConv(input_channle=self.channle_group,
                                                    out_feature_size=self.out_feature_size,
                                                    kernel_size=self.group[i],
                                                    stride=1), )
            if self.nth_layer == 3:
                conv_group = nn.Sequential(UpConv(input_channle=self.channle_group,
                                                    out_feature_size=self.out_feature_size,
                                                    kernel_size=self.group[i],
                                                    stride=2),)
            if self.nth_layer == 4:
                conv_group = nn.Sequential(UpConv(input_channle=self.channle_group,
                                                    out_feature_size=self.channle_group,
                                                    kernel_size=self.group[i],
                                                    stride=2),
                                           UpConv(input_channle=self.channle_group,
                                                  out_feature_size=self.out_feature_size,
                                                  kernel_size=self.group[i],
                                                  stride=2), )
            self.conv_group.append(conv_group)
    def nth_layer_number(self):
        if self.image_size>self.fuse_feature_size:
            nth = self.image_size//self.fuse_feature_size
        else:
            nth = self.fuse_feature_size//self.image_size
        return nth

    def forward(self, x):
        B, C, D, H, W = x.size()
        # print(x.shape)
        #### (num_group,B, channle_group,D,H,W)
        x = x.reshape(B, self.num_group, -1, D, H, W).permute(1, 0, 2, 3, 4, 5)

        for i in range(self.num_group):
            if self.nth_layer == 0:
                xd = self.conv_group[-i](x[i, :])

            if self.nth_layer == 1:
                xd = self.conv_group[-i](x[i, :])

            if self.nth_layer == 2:
                xd = self.conv_group[-i](x[i, :])

            if self.nth_layer == 3:
                xd = self.conv_group[-i](x[i, :])

            if self.nth_layer == 4:
                xd = self.conv_group[-i](x[i, :])


            B, c, d, h, w = xd.size()
            xd = xd.reshape(B, -1, 1, d, h, w)
            if i == 0:
                x_abcd = xd
            else:
                x_abcd = torch.cat([x_abcd, xd], dim=2)

        return x_abcd


if __name__ == "__main__":
    import torch as t
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    print('-----' * 5)
    # rga = t.randn(2, 32, 128, 128, 128)
    # rga = t.randn(2, 256, 16, 16, 16)
    rga = t.randn(2, 256, 16, 16, 16)
    # rga = t.randn(2, 256, 16, 16, 16)
    # rgb = t.randn(1, 3, 352, 480, 150)
    # net = Unet()
    # out = net(rga)
    # print(out.shape)
    # a = t.randn(2, 32, 128, 64, 64)
    net = Mix_channle2(input_size=16,
                       input_channle=256,
                       out_feature_size=32,
                       fuse_feature_size=64,
                       group=[3,5,7,11],
                       image_size=128,
                       nth_layer=4)
    print(net)
    flop = FlopCountAnalysis(net, rga)
    print("flop", flop)
    print("paramr", parameter_count_table(net))
    # print("flop",flop.total())
    out = net(rga)
    # print(net)
    print(out.shape)
