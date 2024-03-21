from functools import reduce, lru_cache
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from moudels.backbones.swintransformer import SwinTransformerSys3D
import copy
from moudels.backbones.ODconv3D import ODConv3d
from moudels.backbones.Dynamic_conv import Dynamic_conv3d
from moudels.backbones.dcd import conv_dy
from torch.nn.functional import interpolate
from moudels.backbones.CMMM import *
def trilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor) * (
            1 - abs(og[2] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :, :] = bilinear_filter
    return torch.from_numpy(weight)


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)

    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD_od(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False, kernel_num=4):
        super(ConvD_od, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = ODConv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, norm=norm,
                              kernel_num=4)
        self.conv2 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, norm=norm,
                              kernel_num=4)

        self.bn1 = normalization(planes, norm)
        self.bn2 = normalization(planes, norm)


    def forward(self, x):
        if not self.first:
            # print("maxpolo",x.shape)
            x = self.maxpool(x)
            # print("maxpolo",x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print("sfae",x1.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        # print("sfae", x2.shape)
        return x


class ConvD_od1(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False, kernel_num=4):
        super(ConvD_od1, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, norm=norm,
                              kernel_num=4)
        # self.conv3 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, kernel_num=4)
        self.bn1 = normalization(planes, norm)
        self.bn2 = normalization(planes, norm)
        # self.bn3 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            # print("maxpolo",x.shape)
            x = self.maxpool(x)
            # print("maxpolo",x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print("sfae",x1.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        # print("sfae", x2.shape)
        return x


class ConvD_dy(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False, kernel_num=4):
        super(ConvD_dy, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = Dynamic_conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = Dynamic_conv3d(planes, planes, kernel_size=3, stride=1, padding=1, )
        # self.conv3 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, kernel_num=4)
        self.bn1 = normalization(planes, norm)
        self.bn2 = normalization(planes, norm)
        # self.bn3 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            # print("maxpolo",x.shape)
            x = self.maxpool(x)
            # print("maxpolo",x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print("sfae",x1.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        # print("sfae", x2.shape)
        return x

class ConvD_dcd(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False, kernel_num=4):
        super(ConvD_dcd, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = conv_dy(inplanes, planes, kernel_size=3, stride=1, padding=1, )
        self.conv2 = conv_dy(planes, planes, kernel_size=3, stride=1, padding=1, )
        # self.conv3 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, kernel_num=4)
        self.bn1 = normalization(planes, norm)
        self.bn2 = normalization(planes, norm)
        # self.bn3 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            # print("maxpolo",x.shape)
            x = self.maxpool(x)
            # print("maxpolo",x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print("sfae",x1.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        # print("sfae", x2.shape)
        return x
class ConvU_od(nn.Module):
    def __init__(self, planes, norm='gn', first=False, kernel_num=4):
        super(ConvU_od, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = ODConv3d(2 * planes, planes, kernel_size=3, stride=1, padding=1,reduction=0.0625, kernel_num=4)
            self.bn1 = normalization(planes, norm)
        self.conv2 = ODConv3d(planes, planes // 2, kernel_size=1, stride=1, padding=0, reduction=0.0625, kernel_num=4)
        self.conv3 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, kernel_num=4)
        self.bn2 = normalization(planes // 2, norm)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class ConvU_od1(nn.Module):
    def __init__(self, planes, norm='gn', first=False, kernel_num=4):
        super(ConvU_od1, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = nn.Conv3d(2 * planes, planes, 3, 1, 1, bias=False)
            self.bn1 = normalization(planes, norm)
        self.conv2 = nn.Conv3d(planes, planes // 2, 1, 1, 0, bias=False)
        self.conv3 = ODConv3d(planes, planes, kernel_size=3, stride=1, padding=1, reduction=0.0625, norm=norm,
                              kernel_num=4)
        self.bn2 = normalization(planes // 2, norm)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class ConvU_dy(nn.Module):
    def __init__(self, planes, norm='gn', first=False, kernel_num=4):
        super(ConvU_dy, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = nn.Conv3d(2 * planes, planes, 3, 1, 1, bias=False)
            self.bn1 = normalization(planes, norm)
        self.conv2 = Dynamic_conv3d(planes, planes // 2, kernel_size=1, stride=1, padding=0, )
        self.conv3 = Dynamic_conv3d(planes, planes, kernel_size=3, stride=1, padding=1, )
        self.bn2 = normalization(planes // 2, norm)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class ConvU_dcd(nn.Module):
    def __init__(self, planes, norm='gn', first=False, kernel_num=4):
        super(ConvU_dcd, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = conv_dy(2 * planes, planes, 3, 1, 1, )
            self.bn1 = normalization(planes, norm)
        self.conv2 = conv_dy(planes, planes // 2, kernel_size=1, stride=1, padding=0, )
        self.conv3 = conv_dy(planes, planes, kernel_size=3, stride=1, padding=1, )
        self.bn2 = normalization(planes // 2, norm)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.bn3(self.conv3(x)))
        return x
class Seg_od(nn.Module):
    def __init__(self, inplanes, num_classes, dropout, norm='gn', kernel_num=4):
        super(Seg_od, self).__init__()
        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = ODConv3d(inplanes, inplanes // 2, kernel_size=3, stride=1, padding=1, reduction=0.0625,
                              kernel_num=4)
        self.conv2 = nn.Conv3d(inplanes // 2, num_classes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = normalization(inplanes // 2, norm)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x
class Seg_dcd(nn.Module):
    def __init__(self, inplanes, num_classes, dropout, norm='gn', kernel_num=4):
        super(Seg_dcd, self).__init__()
        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = conv_dy(inplanes, inplanes // 2, kernel_size=3, stride=1, padding=1, )
        self.conv2 = nn.Conv3d(inplanes // 2, num_classes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = normalization(inplanes // 2, norm)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

class Seg_od1(nn.Module):
    def __init__(self, inplanes, num_classes, dropout, norm='gn', kernel_num=4):
        super(Seg_od1, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = ODConv3d(inplanes, inplanes // 2, kernel_size=3, stride=1, padding=1, reduction=0.0625, norm=norm,
                              kernel_num=4)
        self.conv2 = nn.Conv3d(in_channels=inplanes // 2, out_channels=num_classes, kernel_size=1)
        self.bn1 = normalization(inplanes // 2, norm)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class Seg_dy(nn.Module):
    def __init__(self, inplanes, num_classes, dropout, norm='gn', kernel_num=4):
        super(Seg_dy, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = Dynamic_conv3d(inplanes, inplanes // 2, kernel_size=3, stride=1, padding=1, )
        self.conv2 = nn.Conv3d(in_channels=inplanes // 2, out_channels=num_classes, kernel_size=1)
        self.bn1 = normalization(inplanes // 2, norm)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn3 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            # print("maxpolo",x.shape)
            x = self.maxpool(x)
            # print("maxpolo",x.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # print("sfae",x1.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        # print("sfae", x2.shape)
        return x


class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = nn.Conv3d(2 * planes, planes, 3, 1, 1, bias=False)
            self.bn1 = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes // 2, 1, 1, 0, bias=False)
        self.bn2 = normalization(planes // 2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.cat([prev, x], 1)
        x = self.relu(self.bn3(self.conv3(x)))

        return x


class Seg(nn.Module):
    def __init__(self, inplanes, num_classes, dropout, norm='gn'):
        super(Seg, self).__init__()
        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, inplanes // 2, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = normalization(inplanes // 2, norm)

        self.conv2 = nn.Conv3d(inplanes // 2, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class Swin_Unet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD(c, n, dropout, norm, first=True)
        self.convd1 = ConvD(n, 2 * n, dropout, norm)

        self.convu1 = ConvU(2 * n, norm, True)
        self.seg = Seg(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(64, 64, 64),
            patch_size=(4, 4, 4),
            in_chans=32,
            num_classes=32,
            embed_dim=96,
            depths=[2, 2, 1],
            depths_decoder=[1, 2, 2],
            num_heads=[3, 6, 12],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        # print("x1",x1.shape)
        x = self.convd1(x0)
        y = self.swin(x)
        y = self.convu1(y, x0)
        y = self.seg(y)

        return y

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_2(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD(c, n, dropout, norm, first=True)
        self.convd1 = ConvD(n, 2 * n, dropout, norm)
        self.convd2 = ConvD(2 * n, 4 * n, dropout, norm)

        self.convu2 = ConvU(4 * n, norm, True)
        self.convu1 = ConvU(2 * n, norm, )
        self.seg = Seg(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_chans=64,
            num_classes=64,
            embed_dim=192,
            depths=[2, 1],
            depths_decoder=[1, 2],
            num_heads=[12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        y = self.swin(x2)
        y2 = self.convu2(y, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_3(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_3, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD(c, n, dropout, norm, first=True)  # 4-16 128-128
        self.convd1 = ConvD(n, 2 * n, dropout, norm)  # 16-32 128-64
        self.convd2 = ConvD(2 * n, 4 * n, dropout, norm)  # 32-64   64-32
        self.convd3 = ConvD(4 * n, 8 * n, dropout, norm)  # 64-128 32-16

        self.convu3 = ConvU(8 * n, norm, True)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm, )
        self.seg = Seg(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            in_chans=128,
            num_classes=128,
            embed_dim=384,
            depths=[4, 2],
            depths_decoder=[2, 4],
            num_heads=[12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        y = self.swin(x3)
        y3 = self.convu3(y, x2)
        y2 = self.convu2(y3, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_od_3(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_od_3, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD_od1(c, n, dropout, norm, first=True)  # 4-16 128-128
        self.convd1 = ConvD_od1(n, 2 * n, dropout, norm)  # 16-32 128-64
        self.convd2 = ConvD_od1(2 * n, 4 * n, dropout, norm)  # 32-64   64-32
        self.convd3 = ConvD_od1(4 * n, 8 * n, dropout, norm)  # 64-128 32-16

        self.convu3 = ConvU_od1(8 * n, norm, True)
        self.convu2 = ConvU_od1(4 * n, norm)
        self.convu1 = ConvU_od1(2 * n, norm, )
        self.seg = Seg(2 * n, num_classes, dropout, norm)

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            in_chans=n * 8,
            num_classes=n * 8,
            embed_dim=192,
            depths=[2, 2, 1],
            depths_decoder=[1, 2, 2],
            num_heads=[6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        y = self.swin(x3)
        y3 = self.convu3(y, x2)
        y2 = self.convu2(y3, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")



class Swin_Unet_od_8(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_od_8, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD_od1(c, n, dropout, norm, first=True)  # 4-16 128-128
        self.convd1 = ConvD_od1(n, 2 * n, dropout, norm)  # 16-32 128-64
        self.convd2 = ConvD_od1(2 * n, 4 * n, dropout, norm)  # 32-64   64-32
        self.convd3 = ConvD_od1(4 * n, 8 * n, dropout, norm)  # 64-128 32-16
        self.convd4 = ConvD_od1(8 * n, 16 * n, dropout, norm)  # 128-256 16-8

        self.convu4 = ConvU_od1(16 * n, norm, True)
        self.convu3 = ConvU_od1(8 * n, norm, )
        self.convu2 = ConvU_od1(4 * n, norm, )
        self.convu1 = ConvU_od1(2 * n, norm, )
        self.seg = Seg_od1(2 * n, num_classes, dropout, norm)

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(8, 8, 8),
            patch_size=(4, 4, 4),
            in_chans=n * 16,
            num_classes=n * 16,
            embed_dim=384,
            depths=[2, 1],
            depths_decoder=[1, 2],
            num_heads=[12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)

        # x3= self.convd3(x2)
        y = self.swin(x4)
        y4 = self.convu4(y, x3)
        y3 = self.convu3(y4, x2)
        y2 = self.convu2(y3, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_9(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_9, self).__init__()

        self.convd0 = ConvD_od(c, n, dropout, norm, first=True)
        self.convd1 = ConvD_od(n, 2 * n, dropout, norm)
        self.convd2 = ConvD_od(2 * n, 4 * n, dropout, norm)
        self.convd3 = ConvD_od(4 * n, 8 * n, dropout, norm)

        self.convu3 = ConvU_od(8 * n, norm, True)
        self.convu2 = ConvU_od(4 * n, norm, )
        self.convu1 = ConvU_od(2 * n, norm, )
        self.seg = Seg(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            in_chans=n * 8,
            num_classes=n * 8,
            embed_dim=192,
            depths=[2, 2, 1],
            depths_decoder=[1, 2, 2],
            num_heads=[6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        x3= self.convd3(x2)
        y = self.swin(x3)
        y3 = self.convu3(y, x2)
        y2 = self.convu2(y3, x1)
        y1 = self.convu1(y2, x0)
        y0 = self.seg(y1)

        return y0

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_od_9(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', norm_layer=nn.LayerNorm, num_classes=3, pretrain=None):
        super(Swin_Unet_od_9, self).__init__()

        self.convd0 = ConvD(c, n, dropout, norm, first=True)
        self.convd1 = ConvD_od1(n, 2 * n, dropout, norm)
        self.convd2 = ConvD_od1(2 * n, 4 * n, dropout, norm)

        self.convu2 = ConvU_od1(4 * n, norm, True)
        self.convu1 = ConvU_od1(2 * n, norm, )
        # self.seg = Seg_od1(2*n,num_classes,dropout,norm)
        self.seg = Seg(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_chans=n * 4,
            num_classes=n * 4,
            embed_dim=192,
            depths=[2, 2, 1],
            depths_decoder=[1, 2, 2],
            num_heads=[6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=norm_layer,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        y = self.swin(x2)
        y2 = self.convu2(y, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")

class Swin_Unet_dcd_9(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', norm_layer=nn.LayerNorm, num_classes=3, pretrain=None):
        super(Swin_Unet_dcd_9, self).__init__()

        self.convd0 = ConvD(c, n, dropout, norm, first=True)
        self.convd1 = ConvD_dcd(n, 2 * n, dropout, norm)
        self.convd2 = ConvD_dcd(2 * n, 4 * n, dropout, norm)

        self.convu2 = ConvU_dcd(4 * n, norm, True)
        self.convu1 = ConvU_dcd(2 * n, norm, )
        # self.seg = Seg_od1(2*n,num_classes,dropout,norm)
        self.seg = Seg(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_chans=n * 4,
            num_classes=n * 4,
            embed_dim=192,
            depths=[2, 2, 1],
            depths_decoder=[1, 2, 2],
            num_heads=[6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=norm_layer,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        y = self.swin(x2)
        y2 = self.convu2(y, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")
class Swin_Unet_9_b(nn.Module):
    def __init__(self, c=5, n=32, dropout=0.5, norm='gn', norm_layer=nn.LayerNorm, num_classes=3, pretrain=None):
        super(Swin_Unet_9_b, self).__init__()

        self.convd0 = ConvD(c, n, dropout, norm, first=True)
        self.convd1 = ConvD_dy(n, 2 * n, dropout, norm)
        self.convd2 = ConvD_dy(2 * n, 4 * n, dropout, norm)

        self.convu2 = ConvU_dy(4 * n, norm, True)
        self.convu1 = ConvU_dy(2 * n, norm, )
        self.seg = Seg_dy(2 * n, num_classes, dropout, norm)
        self.last= nn.Conv3d(in_channels=num_classes*2, out_channels=num_classes, kernel_size=1)
        self.seg_up = nn.Conv3d(in_channels=4 * n, out_channels=num_classes, kernel_size=1)
        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_chans=n * 4,
            num_classes=n * 4,
            embed_dim=96,
            depths=[2, 2, 2],
            depths_decoder=[2, 2, 2],
            num_heads=[6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=norm_layer,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x[1:].shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        y = self.swin(x2)   #64
        y2 = self.convu2(y, x1)
        y1 = self.convu1(y2, x0)
        upy = self.seg_up(y)  #3

        y1 = self.seg(y1)
        upy = interpolate(upy,[128,128,128])
        y1 = torch.cat([y1,upy],1)
        y1 = self.last(y1)


        return y1,upy

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_10(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_10, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD(c, n, dropout, norm, first=True)  # 4-16 128-128
        self.convd1 = ConvD(n, 2 * n, dropout, norm)  # 16-32 128-64
        self.convd2 = ConvD(2 * n, 4 * n, dropout, norm)  # 32-64   64-32
        self.convd3 = ConvD(4 * n, 8 * n, dropout, norm)  # 64-128 32-16
        self.convd4 = ConvD(8 * n, 16 * n, dropout, norm)  # 128-256 16-8

        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm, )
        self.convu2 = ConvU(4 * n, norm, )
        self.convu1 = ConvU(2 * n, norm, )
        self.seg = Seg(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(8, 8, 8),
            patch_size=(4, 4, 4),
            in_chans=256,
            num_classes=256,
            embed_dim=192,
            depths=[2, 1],
            depths_decoder=[1, 2],
            num_heads=[6, 12],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)

        # x3= self.convd3(x2)
        y = self.swin(x4)
        y4 = self.convu4(y, x3)
        y3 = self.convu3(y4, x2)
        y2 = self.convu2(y3, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_OD_10(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_OD_10, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD_od(c, n, dropout, norm, first=True)  # 4-16 128-128
        self.convd1 = ConvD_od(n, 2 * n, dropout, norm)  # 16-32 128-64
        self.convd2 = ConvD_od(2 * n, 4 * n, dropout, norm)  # 32-64   64-32
        self.convd3 = ConvD_od(4 * n, 8 * n, dropout, norm, )  # 64-128 32-16
        self.convd4 = ConvD_od(8 * n, 16 * n, dropout, norm, )  # 128-256 16-8

        self.convu4 = ConvU_od(16 * n, norm, True)
        self.convu3 = ConvU_od(8 * n, norm, )
        self.convu2 = ConvU_od(4 * n, norm, )
        self.convu1 = ConvU_od(2 * n, norm, )
        self.seg = Seg_od(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(8, 8, 8),
            patch_size=(4, 4, 4),
            in_chans=512,
            num_classes=512,
            embed_dim=192,
            depths=[2, 1],
            depths_decoder=[1, 2],
            num_heads=[6, 12],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)

        # x3= self.convd3(x2)
        y = self.swin(x4)
        y4 = self.convu4(y, x3)
        y3 = self.convu3(y4, x2)
        y2 = self.convu2(y3, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Swin_Unet_OD_10_1(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=3, pretrain=None):
        super(Swin_Unet_OD_10_1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd0 = ConvD_od1(c, n, dropout, norm, first=True)  # 4-16 128-128
        self.convd1 = ConvD_od1(n, 2 * n, dropout, norm)  # 16-32 128-64
        self.convd2 = ConvD_od1(2 * n, 4 * n, dropout, norm)  # 32-64   64-32
        self.convd3 = ConvD_od1(4 * n, 8 * n, dropout, norm, )  # 64-128 32-16
        self.convd4 = ConvD_od1(8 * n, 16 * n, dropout, norm, )  # 128-256 16-8

        self.convu4 = ConvU_od1(16 * n, norm, True)
        self.convu3 = ConvU_od1(8 * n, norm, )
        self.convu2 = ConvU_od1(4 * n, norm, )
        self.convu1 = ConvU_od1(2 * n, norm, )
        self.seg = Seg_od1(2 * n, num_classes, dropout, norm)
        '''self ensemble
        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)
        '''

        self.swin = SwinTransformerSys3D(
            pretrained=pretrain,
            img_size=(8, 8, 8),
            patch_size=(4, 4, 4),
            in_chans=256,
            num_classes=256,
            embed_dim=192,
            depths=[2, 1],
            depths_decoder=[1, 2],
            num_heads=[6, 12],
            window_size=(7, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            final_upsample="expand_first")
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x0 = self.convd0(x)
        x1 = self.convd1(x0)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)

        # x3= self.convd3(x2)
        y = self.swin(x4)
        y4 = self.convu4(y, x3)
        y3 = self.convu3(y4, x2)
        y2 = self.convu2(y3, x1)
        y1 = self.convu1(y2, x0)
        y1 = self.seg(y1)

        return y1

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


class Unet_od(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=4):
        super(Unet_od, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd1 = ConvD_od1(c, n, dropout, norm, first=True)
        self.convd2 = ConvD_od1(n, 2 * n, dropout, norm)
        self.convd3 = ConvD_od1(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD_od1(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD_od1(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU_od1(16 * n, norm, True)
        self.convu3 = ConvU_od1(8 * n, norm)
        self.convu2 = ConvU_od1(4 * n, norm)
        self.convu1 = ConvU_od1(2 * n, norm)

        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x1 = self.convd1(x)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape)
        x4 = self.convd4(x3)
        # print("x4", x4.shape)
        x5 = self.convd5(x4)
        # print("x5", x5.shape)

        y4 = self.convu4(x5, x4)
        # print("y4", y4.shape)
        y3 = self.convu3(y4, x3)
        # print(y3.shape)
        y2 = self.convu2(y3, x2)
        # print(y2.shape)
        y1 = self.convu1(y2, x1)
        # print(y1.shape)

        y1 = self.seg1(y1)
        # y3 = self.seg3(y3)
        # # print("y3",y3.shape)
        # y_ = self.seg2(y2)
        # # print("y3",y_.shape)
        # y_ = self.upsample(y3)
        # # print("y123",y_.shape)
        # y2 = self.seg2(y2) + self.upsample(y3)
        #
        # y1 = self.seg1(y1) + self.upsample(y2)

        return y1
class Unet_cmmm(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False,mix = True):
        super(Unet_cmmm, self).__init__()
        self.mix = mix
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision
        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        self.seg_bottleneck = nn.Conv3d(n * 16, num_classes, 1)

        self.decoder = nn.ModuleList()
        self.seg =nn.ModuleList()

        for i in range(4):
            decoder = ConvU(n*2**(i+1),norm)
            seg = nn.Conv3d(n*2**(i+1), num_classes, 1)
            if i == 3:
                decoder = ConvU(n * 2 ** (i + 1), norm,True)
            self.decoder.append(decoder)
            self.seg.append(seg)

        if self.mix == True:
            self.mix_c_f = DMAM(image_size=128,
                                input_channle=n,
                                deeps = [1,2,4,8,16], )



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x1 = self.convd1(x)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape)
        x4 = self.convd4(x3)
        # print("x4", x4.shape)
        x5 = self.convd5(x4)
        # print("x5", x5.shape)
        if self.mix==True:
            x_mix = self.mix_c_f([x1,x2,x3,x4,x5])
        else:
            x_mix = [x1,x2,x3,x4,x5]
        # print(len(x_mix))
        seg5 = self.seg_bottleneck(x_mix[-1])
        y= []
        out  = [seg5]
        for i in range(4):
            # print(x_mix[-1].shape,"dddddd",x_mix[-(i+2)].shape)
            if i == 0 :
                y1 = self.decoder[-(i+1)](x_mix[-1],x_mix[-2])
            else:
                y1 = self.decoder[-(i+1)](y[-1], x_mix[-(i+2)])
            y.append(y1)
            out1 = self.seg[-(i+1)](y1)
            out.append(out1)


        if self.training and self.deep_supervision:
            out_all = [out[-1]]
            for i in range(len(out)-1):
                out[i]=F.interpolate(out[i], out[-1].shape[2:],mode='trilinear')
                out_all.append(out[i])

            return torch.stack(out_all, dim=1)


        return out[-1]

class Unet_dcd(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=4):
        super(Unet_dcd, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convd1 = ConvD_dcd(c, n, dropout, norm, first=True)
        self.convd2 = ConvD_dcd(n, 2 * n, dropout, norm)
        self.convd3 = ConvD_dcd(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD_dcd(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD_dcd(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU_dcd(16 * n, norm, True)
        self.convu3 = ConvU_dcd(8 * n, norm)
        self.convu2 = ConvU_dcd(4 * n, norm)
        self.convu1 = ConvU_dcd(2 * n, norm)

        self.seg3 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg2 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        x1 = self.convd1(x)
        # print("x1",x1.shape)
        x2 = self.convd2(x1)
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape)
        x4 = self.convd4(x3)
        # print("x4", x4.shape)
        x5 = self.convd5(x4)
        # print("x5", x5.shape)

        y4 = self.convu4(x5, x4)
        # print("y4", y4.shape)
        y3 = self.convu3(y4, x3)
        # print(y3.shape)
        y2 = self.convu2(y3, x2)
        # print(y2.shape)
        y1 = self.convu1(y2, x1)
        # print(y1.shape)

        y1 = self.seg1(y1)
        # y3 = self.seg3(y3)
        # # print("y3",y3.shape)
        # y_ = self.seg2(y2)
        # # print("y3",y_.shape)
        # y_ = self.upsample(y3)
        # # print("y123",y_.shape)
        # y2 = self.seg2(y2) + self.upsample(y3)
        #
        # y1 = self.seg1(y1) + self.upsample(y2)

        return y1


if __name__ == "__main__":
    import torch as t
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    print('-----' * 5)
    rga = t.randn(2, 5, 128, 128, 128)
    # rgb = t.randn(1, 3, 352, 480, 150)
    # net = Unet()
    # out = net(rga)
    # print(out.shape)
    # a = t.randn(2, 32, 128, 64, 64)
    net = Unet_cmmm(c=5, n=32, dropout=0.5, norm='gn', num_classes=3,deep_supervision=True)
    print(net)
    flop = FlopCountAnalysis(net, rga)
    print("flop",flop)
    print("paramr", parameter_count_table(net))
    # print("flop",flop.total())
    # net =BasicLayer(
    #     dim=
    # )
    # net = PatchEmbed3D(img_size=(64,64,64),
    #                    patch_size=(4,4,4),
    #                    in_chans=256,
    #                    embed_dim=512,
    #                    norm_layer=nn.LayerNorm)
    out = net(rga)
    # print(net)
    print(out[0].shape,out[1].shape)
