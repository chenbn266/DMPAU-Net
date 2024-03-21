import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm



class Mix_channle(nn.Module):
    def __init__(self,
                 input_size: int = 128,
                 input_channle: int = 24,
                 out_feature_size: int = 24,
                 fuse_feature_size: int = 64,
                 group: Sequence[int] = [1, 3, 5, 7],
                 image_size: int = 128, ):
        super(Mix_channle, self).__init__()
        self.input_size = int(input_size)
        self.group = group
        self.num_group = len(group)
        self.channle_group = input_channle // self.num_group
        self.out_feature_size = out_feature_size
        self.fuse_feature_size = fuse_feature_size
        self.image_size = int(image_size)
        self.conv_group = nn.ModuleList()
        for i in range(self.num_group):
            if self.input_size < group[i]:
                assert self.input_size % 2 == 0, "input_size must be even number"
                group[i] = self.input_size - 1
                padding = group[i] // 2
            else:
                group[i] = group[i]
                padding = i
            print(self.group[i],padding)
            conv_group = nn.Sequential(nn.Conv3d(self.channle_group, self.channle_group,
                                                 kernel_size=self.group[i],
                                                 stride=1,
                                                 padding=padding,
                                                 groups=self.channle_group),
                                       nn.Conv3d(self.channle_group, out_feature_size,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 groups=out_feature_size),
                                       nn.InstanceNorm3d(out_feature_size),
                                       nn.LeakyReLU())

            self.conv_group.append(conv_group)
            self.pool = nn.AdaptiveAvgPool3d((self.fuse_feature_size, self.fuse_feature_size, self.fuse_feature_size))

    def forward(self, x):
        B, C, D, H, W = x.size()
        # print(x.shape,self.num_group)
        x = x.reshape(B, C // self.channle_group, self.channle_group, D, H, W).permute(1, 0, 2, 3, 4, 5)
        # print(x.shape, self.num_group,"fffffff")
        for i in range(self.num_group):
            xd = self.conv_group[-i](x[i, :])
            # print(self.image_size)
            if D != self.image_size // 2:
                if D > self.image_size // 2:
                    xd = self.pool(xd)
                else:
                    xd = F.interpolate(xd, [self.fuse_feature_size, self.fuse_feature_size, self.fuse_feature_size],
                                       mode='trilinear', )
                    # print("uping",xd.shape)
            xd = xd.reshape(B, self.out_feature_size, -1, self.fuse_feature_size, self.fuse_feature_size,
                            self.fuse_feature_size)
            if i == 0:
                x_abcd = xd
            else:
                x_abcd = torch.cat([x_abcd, xd], dim=2)

        return x_abcd


class DMAM(nn.Module):
    def __init__(self,
                 image_size: int = 128,
                 input_channle: int = 16,
                 deeps: Sequence[int] = [1, 2, 4, 8],
                 group: Sequence[int] = [1, 3, 5, 7],
                 ):
        super(DMAM, self).__init__()

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

        for i in range(self.num_deeps):
            split_group = Mix_channle(input_size=self.image_size // (2 ** i),
                                      input_channle=self.deeps[i] * self.input_channle,
                                      out_feature_size=self.num_patch,
                                      fuse_feature_size=self.fuse_size,)

            self.split_group.append(split_group)
            fuse2 = nn.Sequential(
                nn.Conv3d(self.num_group * self.num_deeps * self.num_patch,
                          self.deeps[i] * self.input_channle, 1, 1, 0),
                nn.InstanceNorm3d(self.deeps[i] * self.input_channle),
                nn.LeakyReLU()
            )
            self.fuse2.append(fuse2)
        self.pool = nn.AdaptiveAvgPool3d(self.fuse_shape)
        self.fuse = nn.Sequential(nn.Conv3d(self.num_group * self.num_deeps, self.num_group * self.num_deeps, 1, 1, 0),
                                  nn.InstanceNorm3d(self.num_group * self.num_deeps),
                                  nn.LeakyReLU(inplace=True))

    def forward(self, x):

        for i in range(self.num_deeps):
            # print(i,)
            B, C, _, _, _ = x[i].shape
            # print(i, x[i][:,0:C//2,:].shape)
            x_abcd = self.split_group[i](x[i])
            B, channle_group, num_group, D, H, W = x_abcd.shape
            print(x_abcd.shape)
            if i == 0:
                x_all = x_abcd
            else:
                x_all = torch.cat([x_all, x_abcd], dim=2)

        for i in range(self.num_patch):
            x_all[:, i, :] = self.fuse(x_all[:, i, :].clone())

        x_all = x_all.reshape(B, -1, D, H, W)
        for i in range(self.num_deeps):
            # print(x_all.shape,"x-all")
            x_all_r = self.fuse2[i](x_all)
            # print(x_all_r.shape, "x-all")
            _, _, d_x_all_size, h_x_all_size, w_x_all_size = x_all_r.shape
            _, C, d_xi_size, h_xi_size, w_xi_size = x[i].shape
            if d_x_all_size > d_xi_size:
                x_k = F.interpolate(x_all_r, x[i].shape[2:], mode='trilinear', )
            else:
                x_k = F.adaptive_avg_pool3d(x_all_r, x[i].shape[2:])

            # print(x_k.shape,x[i][:,0:C//2,:].shape,"dddddddd")
            # y[i] = torch.cat([x_k,x[i][:,C//2:C,:]],dim=1)
            x[i] = x_k + x[i]  # res
            # x[i] = x_k
            # print(x[i].shape)

        return x


class mix_channle_fuse2(nn.Module):
    def __init__(self,
                 image_size: int = 128,
                 input_channle: int = 16,
                 deeps: Sequence[int] = [1, 2, 4, 8, 16],
                 group: Sequence[int] = [3,5,7,11],
                 kernel_size: int = 3,
                 fuse_to_size:int =64,
                 mid_channle:int=64,
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
        self.mid_channle = mid_channle

        for i in range(self.num_deeps):
            split_group = Mix_channle2(input_size=self.image_size // (2 ** i),
                                       input_channle=self.deeps[i] * self.input_channle,
                                       out_feature_size=self.mid_channle,
                                       fuse_feature_size=self.fuse_size,
                                       nth_layer=i,
                                       fuse_to_size=fuse_to_size)
            self.split_group.append(split_group)
            if fuse_to_size ==32:
                if i == 0:
                    fuse2 = nn.Sequential(UpConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                 out_feature_size=(self.deeps[i] + 1) * self.input_channle,
                                                 kernel_size=kernel_size,
                                                 stride=2),
                                          UpConv(input_channle=(self.deeps[i] + 1) * self.input_channle,
                                                 out_feature_size=self.deeps[i] * self.input_channle,
                                                 kernel_size=kernel_size,
                                                 stride=2), )

                if i == 1:
                    fuse2 = nn.Sequential(UpConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                 out_feature_size=self.deeps[i] * self.input_channle,
                                                 kernel_size=kernel_size,
                                                 stride=2), )
                if i == 2:
                    fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                   out_feature_size=self.deeps[i] * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=1), )
                if i == 3:
                    fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                   out_feature_size=self.deeps[i] * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2), )
                if i == 4:
                    fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                   out_feature_size=(self.deeps[i] - 1) * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2),
                                          DownConv(input_channle=(self.deeps[i] - 1) * self.input_channle,
                                                   out_feature_size=self.deeps[i] * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2), )
                self.fuse2.append(fuse2)
            if fuse_to_size==64:
                if i == 0:
                    fuse2 = nn.Sequential(UpConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                 out_feature_size=self.deeps[i] * self.input_channle,
                                                 kernel_size=kernel_size,
                                                 stride=2),)

                if i == 1:
                    fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                 out_feature_size=self.deeps[i] * self.input_channle,
                                                 kernel_size=kernel_size,
                                                 stride=1), )
                if i == 2:
                    fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                   out_feature_size=self.deeps[i] * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2), )
                if i == 3:
                    fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                   out_feature_size=(self.deeps[i] - 1) * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2),
                                          DownConv(input_channle=(self.deeps[i] - 1) * self.input_channle,
                                                   out_feature_size=self.deeps[i] * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2),
                                          )
                if i == 4:
                    fuse2 = nn.Sequential(DownConv(input_channle=self.mid_channle * self.num_deeps * self.num_group,
                                                   out_feature_size=(self.deeps[i] - 2) * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2),
                                          DownConv(input_channle=(self.deeps[i] - 2) * self.input_channle,
                                                   out_feature_size=(self.deeps[i] - 1) * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2),
                                          DownConv(input_channle=(self.deeps[i] - 1) * self.input_channle,
                                                   out_feature_size=self.deeps[i] * self.input_channle,
                                                   kernel_size=kernel_size,
                                                   stride=2),)
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
                # print(x_all.shape,x_abcd.shape,i)
                x_all = torch.cat([x_all, x_abcd], dim=2)

        for i in range(self.num_patch):
            x_all[:, i, :] = self.fuse(x_all[:, i, :].clone())
        x_all = x_all.reshape(B, -1,D, H, W)
        print(x_all.shape)
        for i in range(self.num_deeps):
            x_all_r = self.fuse2[i](x_all)
            print(x_all_r.shape)
            y = x1[i]
            # print(x_all_r.shape,i,"dd")
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

class Mix_channle2(nn.Module):
    def __init__(self,
                 input_size: int = 128,
                 input_channle: int = 24,
                 out_feature_size: int = 24,
                 fuse_feature_size: int = 64,
                 group: Sequence[int] = [1, 3, 5, 7],
                 image_size: int = 128,
                 nth_layer: int = 0,
                 fuse_to_size: int =64,):
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
        for i in range(self.num_group):
            if fuse_to_size==32:
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

            if fuse_to_size ==64:
                if self.nth_layer == 0:
                    conv_group = nn.Sequential(DownConv(input_channle=self.channle_group,
                                                        out_feature_size=self.out_feature_size,
                                                        kernel_size=self.group[i],
                                                        stride=2),
                                                )
                if self.nth_layer == 1:
                    conv_group = nn.Sequential(DownConv(input_channle=self.channle_group,
                                                        out_feature_size=self.out_feature_size,
                                                        kernel_size=self.group[i],
                                                        stride=1), )
                if self.nth_layer == 2:
                    conv_group = nn.Sequential(UpConv(input_channle=self.channle_group,
                                                        out_feature_size=self.out_feature_size,
                                                        kernel_size=self.group[i],
                                                        stride=2), )
                if self.nth_layer == 3:
                    conv_group = nn.Sequential(UpConv(input_channle=self.channle_group,
                                                      out_feature_size=self.channle_group,
                                                      kernel_size=self.group[i],
                                                      stride=2),
                                               UpConv(input_channle=self.channle_group,
                                                      out_feature_size=self.out_feature_size,
                                                      kernel_size=self.group[i],
                                                      stride=2),  )
                if self.nth_layer == 4:
                    conv_group = nn.Sequential(UpConv(input_channle=self.channle_group,
                                                      out_feature_size=self.channle_group,
                                                      kernel_size=self.group[i],
                                                      stride=2),
                                               UpConv(input_channle=self.channle_group,
                                                      out_feature_size=self.out_feature_size,
                                                      kernel_size=self.group[i],
                                                      stride=2),
                                               UpConv(input_channle=self.out_feature_sizep,
                                                      out_feature_size=self.out_feature_size,
                                                      kernel_size=self.group[i],
                                                      stride=2),)
                self.conv_group.append(conv_group)

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
            print(xd.size)
            xd = xd.reshape(B, -1, 1, d, h, w)
            if i == 0:
                x_abcd = xd
            else:
                x_abcd = torch.cat([x_abcd, xd], dim=2)

        return x_abcd

class mix_channle_fuse3(nn.Module):
    def __init__(self,
                 image_size: int = 128,
                 input_channle: int = 16,
                 deeps: Sequence[int] = [1, 2, 4, 8],
                 group: Sequence[int] = [3,5,7,11],
                 up_size: int = 3,
                 ):
        super(mix_channle_fuse3, self).__init__()

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
            print(x_all_r.shape)
            y = x1[i]

            x1[i] =self.fuse3[i](torch.cat([x_all_r , y],dim=1))  # res

        return x




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
    net = mix_channle_fuse2(image_size=128,
                                             input_channle=16,
                                             deeps=[1, 2, 4, 8],
                                             group=[3,5,7,11],
                                             kernel_size=7,
                                             fuse_to_size=32,
                                             mid_channle=32)
    print(net)
    flop = FlopCountAnalysis(net, rga)
    print("flop", flop)
    print("paramr", parameter_count_table(net))
    # print("flop",flop.total())
    out = net(rga)
    # print(net)
    print(out.shape)
