import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from moudels.isnet.imagelevel import *

from fvcore.nn import FlopCountAnalysis, parameter_count_table
from moudels.backbones.swintransformer import SwinTransformerSys3D,SwinTransformerSys3D_encoder
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


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True,negative_slope= 0.01)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1,
        #               bias=False, groups=inplanes),
        #     nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, padding=0,
        #               bias=False, groups=1),
        # )
        self.bn1 = normalization(planes, norm)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # self.conv2 = nn.Sequential(
        #                            nn.Conv3d(planes,planes, kernel_size=3, stride=1, padding=1,
        #                                      bias=False, groups=planes),
        #                            nn.Conv3d(planes , planes, kernel_size=1, stride=1, padding=0,
        #                                      bias=False, groups=1),
        #                            )
        self.bn2 = normalization(planes, norm)

        # self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # self.bn3 = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            #print("maxpolo",x.shape)
            x = self.maxpool(x)
            #print("maxpolo",x.shape)
        x1 = self.relu(self.bn1(self.conv1(x)))
        #print("sfae",x1.shape)
        x2 = self.relu(self.bn2(self.conv2(x1)))
        #print("sfae", x2.shape)
        return x2
class lk_conv(nn.Module):
    def __int__(self,inplanes , planes, kernl_size=7,padding=3):
        super(lk_conv,self).__int__()
        self.convh = nn.Conv3d(inplanes, planes, kernel_size=(kernl_size,1,1), padding=(padding,1,1), stride=1, bias=False)
        self.convw = nn.Conv3d(inplanes, planes, kernel_size=(1, kernl_size, 1), padding=(1, padding, 1), stride=1,
                               bias=False)
        self.convd = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, kernl_size), padding=(1, 1, padding), stride=1,
                               bias=False)
        self.bn = normalization(planes,"gn")
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
    def forward(self,x):
        h = self.convh(x)
        h = self.relu(self.bn(h))
        w = self.convw(x)
        w = self.relu(self.bn(w))
        d = self.convd(x)
        d = self.relu(self.bn(d))
        x = h+w+d+x
        return x
class sub2(nn.Module):
    def __init__(self, inplanes, ):
        super(sub2, self).__init__()
        # self.gn = nn.GroupNorm(num_channels=inplanes,num_groups=4)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        # self.pool = nn.AvgPool3d(2,2)

    def forward(self,x):

        pool = self.pool(x)
        weights2 = F.softmax(pool,dim=1)
        pool = F.interpolate(pool,x.size()[2:],mode='trilinear')
        sub = x-pool
        # weights = torch.sigmoid(sub)
        weights = F.softmax(sub,dim=1)

        x1 = x*weights*weights2
        x =  x1+x
        #
        return x
class sub(nn.Module):
    def __init__(self, inplanes, ):
        super(sub, self).__init__()
        # self.gn = nn.GroupNorm(num_channels=inplanes,num_groups=4)
        # self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.pool = nn.AvgPool3d(2,2)

    def forward(self,x):

        pool = self.pool(x)

        pool = F.interpolate(pool,scale_factor=2,mode='trilinear')
        sub = x-pool
        # weights = torch.sigmoid(sub)
        weights = F.softmax(sub,dim=1)
        x1 = x*weights
        x =  x1+x
        #
        return x
class ConvD_lk(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD_lk, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.LeakyReLU(inplace=True,negative_slope= 0.01)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1, bias=False)

        self.bn1 = normalization(planes, norm)
        self.gn = nn.GroupNorm(num_channels=planes, num_groups=4)

        # self.conv1 = nn.Conv3d(in_channels=inplanes * 2, out_channels=planes, kernel_size=3, padding=1, stride=1)


        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = normalization(planes, norm)
        self.sub = sub(planes)


    def forward(self, input):
        x=input
        if not self.first:
            #print("maxpolo",x.shape)
            x = self.maxpool(x)
            #print("maxpolo",x.shape)
        x1 = self.relu(self.bn1(self.conv1(x)))


        # x2 = self.sub(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(self.bn2(x2))
        x2 = self.sub(x2)
        #print("sfae", x2.shape)
        return x2
class ConvU_sw(nn.Module):
    def __init__(self, planes,embed_dim, norm='gn', first=False):
        super(ConvU_sw, self).__init__()

        self.first = first
        self.relu = nn.LeakyReLU(inplace=True)

        if not self.first:
            self.conv1 = nn.Conv3d(2 * planes, planes, 3, 1, 1, bias=False)
            self.bn1 = normalization(planes, norm)
        self.conv2 = nn.Conv3d(planes, planes // 2, 1, 1, 0, bias=False)
        # else:
        #     self.conv2 = nn.Conv3d(embed_dim, planes // 2, 1, 1, 0, bias=False)
        self.bn2 = normalization(planes // 2, norm)

        self.conv3 = nn.Conv3d(planes//2+embed_dim//2, planes, 3, 1, 1, bias=False)
        self.bn3 = normalization(planes, norm)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))
            # print("upx",x.shape)
        # print("upqian",x.shape,prev.shape)
        y = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # print("uphou", y.shape)
        y = self.relu(self.bn2(self.conv2(y)))
        # print("upyy", y.shape)
        y = torch.cat([prev, y], 1)
        # print("upc", y.shape)
        y = self.relu(self.bn3(self.conv3(y)))
        # print("upend", y.shape)

        return y
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
            # print("upx",x.shape)
        # print("upqian",x.shape,prev.shape)
        y = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # print("uphou", y.shape)
        y = self.relu(self.bn2(self.conv2(y)))
        # print("upyy", y.shape)
        y = torch.cat([prev, y], 1)
        # print("upc", y.shape)
        y = self.relu(self.bn3(self.conv3(y)))
        # print("upend", y.shape)

        return y

class Unet_en(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_en, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD_lk(n, 2 * n, dropout, norm)
        self.convd3 = ConvD_lk(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD_lk(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD_lk(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)

        self.seg_pred = nn.Conv3d(16 * n, num_classes, 1)
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
        # print("x4"
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)
        # print(x5.shape)
        # il = self.ilc(x5)
        # pred = self.seg_pred(x5)
        # sl = self.slc(x5, pred)
        # sl,pred = self.slc(x5)    #v18
        # print(sl.shape)
        y4 = self.convu4(x5, x4)
        # print("y4", y4.shape)
        y3 = self.convu3(y4, x3)
        # print(y3.shape)
        y2 = self.convu2(y3, x2)
        # print(y2.shape)
        y1 = self.convu1(y2, x1)
        # print(y1.shape)


        out = self.seg1(y1)


        # if self.training and self.deep_supervision:
        #     out_all = [out]
            # pred = F.interpolate(pred, out.shape[2:])
            # out_all.append(pred)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            # return torch.stack(out_all, dim=1)
        return out
class Unet(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)
        # ilc_cfg = {
        #     'feats_channels': 16*n,
        #     'transform_channels': 8*n,
        #     'concat_input': True,
        #     'norm_cfg': {'type': 'InstanceNorm3d'},
        #     'act_cfg': {'type': 'ReLU', 'inplace': True},
        #     'align_corners': False,
        # }
        slc_cfg = {'feats_channels': 16*n,
                   'transform_channels': 8*n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   #'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True},}

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = SemanticLevelContext_3D_45(**slc_cfg)

        self.seg_pred = nn.Conv3d(16 * n, num_classes, 1)
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
        # print("x4"
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)
        # print(x5.shape)
        # il = self.ilc(x5)
        # pred = self.seg_pred(x5)
        # sl = self.slc(x5, pred)
        sl,pred = self.slc(x5)    #v18
        # print(sl.shape)
        y4 = self.convu4(sl, x4)
        # print("y4", y4.shape)
        y3 = self.convu3(y4, x3)
        # print(y3.shape)
        y2 = self.convu2(y3, x2)
        # print(y2.shape)
        y1 = self.convu1(y2, x1)
        # print(y1.shape)


        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred = F.interpolate(pred, out.shape[2:])
            out_all.append(pred)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out

######################################v63
class Unet_2(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)
        # ilc_cfg = {
        #     'feats_channels': 16*n,
        #     'transform_channels': 8*n,
        #     'concat_input': True,
        #     'norm_cfg': {'type': 'InstanceNorm3d'},
        #     'act_cfg': {'type': 'ReLU', 'inplace': True},
        #     'align_corners': False,
        # }

        slc_cfg = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = PAM(**slc_cfg)

        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc34 = PAM(**slc_cfg34)

        slc_cfg2 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc2 = PAM(**slc_cfg2)

        self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)
        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+16 * n + 8 * n, 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.seg_pred34 = nn.Conv3d(8*n, num_classes, 1)
        self.seg_pred2 = nn.Conv3d(4 * n, num_classes, 1)
        # self.conv12 = nn.Sequential(nn.Conv3d(8 * n + 8 * n + 4 * n, 4 * n, 3, 1, 1),
        #                             normalization(8 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
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
        # print("x4"
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)
        # print(x5.shape)
        # il = self.ilc(x5)
        pred5 = self.seg_pred5(x5)
        # sl = self.slc(x5, pred)
        sl = self.slc(x5,pred5)    #v18
        # print(sl.shape)


        # y4,pred4 = self.convu4(sl,x4,pred5)
        # y3,pred3= self.convu3(y4, x3,pred4)
        # y2,pred2 = self.convu2(y3, x2,pred3)


        # y1 ,pred1= self.convu1(y2, x1,pred2)
        # print(y1.shape)
        y4 = self.convu4(sl, x4)
        y3 = self.convu3(y4, x3)

        y4 =  F.interpolate(y4, scale_factor=2, mode='trilinear', align_corners=False)
        sl = F.interpolate(sl,y3.shape[2:],mode='trilinear')
        y3 = self.conv34(torch.concat([y3,y4,sl],dim=1))
        pred34 = self.seg_pred34(y3)
        sl34 = self.slc34(y3,pred34)
        y2 = self.convu2(sl34, x2)

        pred12 = self.seg_pred12(y2)
        sl12 = self.slc2(y2, pred12)

        y1 = self.convu1(sl12, x1)

        # sl34 = F.interpolate(sl34,y1.shape[2:],mode='trilinear')
        # y2 = F.interpolate(y2, y1.shape[2:], mode='trilinear')
        # y1 = self.conv34(torch.concat([y1, y2, sl34], dim=1))
        # pred12 = self.seg_pred12(y1)
        # sl12 = self.slc12(y1, pred12)
        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred12, out.shape[2:])
            pred2 = F.interpolate(pred34, out.shape[2:],mode='trilinear')
            pred3= F.interpolate(pred5, out.shape[2:],mode='trilinear')
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out
class Unet_64(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_64, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        # self.convu1 = ConvU(2 * n, norm)
        self.convu1 = nn.Sequential(nn.Conv3d(4*n, n, 3, 1, 1),
                                    normalization(n, norm),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                                    nn.Conv3d(n, n, 3, 1, 1),
                                    normalization(n, norm),
                                    nn.LeakyReLU(inplace=True),
                                    )


        slc_cfg = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = PAM(**slc_cfg)

        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc34 = PAM(**slc_cfg34)

        slc_cfg2 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc2 = PAM(**slc_cfg2)
        # slc_cfg1 = {'feats_channels': n,
        #             'out_feats_channels':  3,
        #             'concat_input': True,
        #             'norm_cfg': {'type': 'InstanceNorm3d'},
        #             # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
        #             'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        # self.slc1 = SemanticLevelContext_3D_45_d3(**slc_cfg1)
        self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)
        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+16 * n + 8 * n, 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.seg_pred34 = nn.Conv3d(8*n, num_classes, 1)
        self.seg_pred2 = nn.Conv3d(4 * n, num_classes, 1)
        # self.conv12 = nn.Sequential(nn.Conv3d(8 * n + 8 * n + 4 * n, 4 * n, 3, 1, 1),
        #                             normalization(8 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
        self.seg1 = nn.Conv3d( n, num_classes, 1)

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
        # print("x4"
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)
        # print(x5.shape)
        # il = self.ilc(x5)
        pred5 = self.seg_pred5(x5)
        # sl = self.slc(x5, pred)
        sl = self.slc(x5,pred5)    #v18
        # print(sl.shape)


        # y4,pred4 = self.convu4(sl,x4,pred5)
        # y3,pred3= self.convu3(y4, x3,pred4)
        # y2,pred2 = self.convu2(y3, x2,pred3)


        # y1 ,pred1= self.convu1(y2, x1,pred2)
        # print(y1.shape)
        y4 = self.convu4(sl, x4)
        y3 = self.convu3(y4, x3)

        y4 =  F.interpolate(y4, scale_factor=2, mode='trilinear', align_corners=False)
        sl = F.interpolate(sl,y3.shape[2:],mode='trilinear')
        y3 = self.conv34(torch.concat([y3,y4,sl],dim=1))
        pred34 = self.seg_pred34(y3)
        sl34 = self.slc34(y3,pred34)
        y2 = self.convu2(sl34, x2)

        pred12 = self.seg_pred2(y2)
        sl2 = self.slc2(y2, pred12)

        y1 = self.convu1(sl2)
        # y1 = F.interpolate(y1,x.shape[2:],mode='trilinear')

        # sl34 = F.interpolate(sl34,y1.shape[2:],mode='trilinear')
        # y2 = F.interpolate(y2, y1.shape[2:], mode='trilinear')
        # y1 = self.conv34(torch.concat([y1, y2, sl34], dim=1))
        # pred12 = self.seg_pred12(y1)
        # sl12 = self.slc12(y1, pred12)
        out = self.seg1(y1)
        out = self.slc1(y1,out)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred12, out.shape[2:])
            pred2 = F.interpolate(pred34, out.shape[2:],mode='trilinear')
            pred3= F.interpolate(pred5, out.shape[2:],mode='trilinear')
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out


class Unet_74_CMMM(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_74_CMMM, self).__init__()

        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)

        self.mix = DMAM(image_size=128,
                        input_channle=n,
                        deeps=[1, 2, 4, 8, 16],
                        )

        slc_cfg = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = PAM(**slc_cfg)
        slc_cfg45 = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc45 = PAM(**slc_cfg45)
        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc34 = PAM(**slc_cfg34)

        slc_cfg23 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc23 = PAM(**slc_cfg23)

        self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)

        self.conv45 = nn.Sequential(nn.Conv3d(16 * n *2, 16 * n, 3, 1, 1),
                                    normalization(16 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred45 = nn.Conv3d(16 * n, num_classes, 1)
        self.seg_pred45_2 = nn.Conv3d(num_classes * 2, num_classes, 1)


        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+8*n , 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.seg_pred34 = nn.Conv3d(8*n, num_classes, 1)
        self.seg_pred34_2 = nn.Conv3d(num_classes*2, num_classes, 1)
        self.conv23 = nn.Sequential(nn.Conv3d(4 * n + 8 * n, 4 * n, 3, 1, 1),
                                    normalization(4 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred23 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg_pred23_2 = nn.Conv3d(num_classes*2, num_classes, 1)
        # self.conv12 = nn.Sequential(nn.Conv3d(8 * n + 8 * n + 4 * n, 4 * n, 3, 1, 1),
        #                             normalization(8 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        self.upconv5 = nn.Conv3d(16*n,16*n,1)
        self.upconv4 = nn.Conv3d(16*n,16*n, 1)
        self.upconv3 = nn.Conv3d(8*n, 8*n, 1)

        # self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)
        # self.seg_pred3 = nn.Conv3d(8 * n, num_classes, 1)
        # self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) :
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
        # print("x4"
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)

        x1,x2,x3,x4,x5 = self.mix([x1,x2,x3,x4,x5])

        pred5 = self.seg_pred5(x5)
        # sl = self.slc(x5, pred)
        sl = self.slc(x5,pred5)    #v18
        y4 = self.convu4(sl, x4)
        sl = F.interpolate(sl,y4.shape[2:],mode='trilinear')
        pred5 = F.interpolate(pred5,y4.shape[2:],mode='trilinear')
        sl = self.upconv5(sl)
        y4 = torch.concat([y4,sl],dim=1)

        y45 = self.conv45(y4)
        pred45 = self.seg_pred45(y45)
        pred45 = self.seg_pred45_2(torch.concat([pred45,pred5],dim=1))
        sl45 = self.slc45(y45,pred45)

        y3 = self.convu3(sl45, x3)

        # y45 =  F.interpolate(y45, scale_factor=2, mode='trilinear', align_corners=False)
        sl45 = F.interpolate(sl45,y3.shape[2:],mode='trilinear')
        pred45 = F.interpolate(pred45, y3.shape[2:], mode='trilinear')
        sl45 = self.upconv4(sl45)
        y3 = self.conv34(torch.concat([y3,sl45],dim=1))
        pred34 = self.seg_pred34(y3)
        pred34 = self.seg_pred34_2(torch.concat([pred34, pred45], dim=1))
        sl34 = self.slc34(y3,pred34)

        y2 = self.convu2(sl34, x2)
        sl34 = F.interpolate(sl34, y2.shape[2:], mode='trilinear')
        pred34 = F.interpolate(pred34, y2.shape[2:], mode='trilinear')
        sl34 = self.upconv3(sl34)
        y2 = self.conv23(torch.concat([y2, sl34], dim=1))
        pred23 = self.seg_pred23(y2)
        pred23 = self.seg_pred23_2(torch.concat([pred23, pred34], dim=1))
        sl23 = self.slc23(y2, pred23)

        y1 = self.convu1(sl23, x1)
        #
        # y4 = self.convu4(x5, x4)
        # # print("y4", y4.shape)
        # y3 = self.convu3(y4, x3)
        # # print(y3.shape)
        # y2 = self.convu2(y3, x2)
        # # print(y2.shape)
        # y1 = self.convu1(y2, x1)
        # # print(y1.shape)
        #
        # seg_pred5 = self.seg_pred5(x5)
        # seg_pred3 = self.seg_pred3(y3)
        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred45, out.shape[2:],mode='trilinear')
            pred2 = F.interpolate(pred34, out.shape[2:],mode='trilinear')
            pred3= F.interpolate(pred23, out.shape[2:],mode='trilinear')
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out
class Unet_74_mc(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_74_mc, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)
        # ilc_cfg = {
        #     'feats_channels': 16*n,
        #     'transform_channels': 8*n,
        #     'concat_input': True,
        #     'norm_cfg': {'type': 'InstanceNorm3d'},
        #     'act_cfg': {'type': 'ReLU', 'inplace': True},
        #     'align_corners': False,
        # }

        slc_cfg = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = PAM(**slc_cfg)
        slc_cfg45 = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc45 = PAM(**slc_cfg45)
        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc34 = PAM(**slc_cfg34)

        slc_cfg23 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc23 = PAM(**slc_cfg23)

        # slc_cfg12 = {'feats_channels': 2 * n,
        #              'out_feats_channels': 2 * n,
        #              'concat_input': True,
        #              'norm_cfg': {'type': 'InstanceNorm3d'},
        #              # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
        #              'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        # self.slc12 = SemanticLevelContext_3D_45_d3(**slc_cfg12)

        self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)

        self.conv45 = nn.Sequential(nn.Conv3d(16 * n *2, 16 * n, 3, 1, 1),
                                    normalization(16 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred45 = nn.Conv3d(16 * n, num_classes, 1)
        self.seg_pred45_2 = nn.Conv3d(num_classes * 2, num_classes, 1)


        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+8*n , 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.seg_pred34 = nn.Conv3d(8*n, num_classes, 1)
        self.seg_pred34_2 = nn.Conv3d(num_classes*2, num_classes, 1)
        self.conv23 = nn.Sequential(nn.Conv3d(4 * n + 8 * n, 4 * n, 3, 1, 1),
                                    normalization(4 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred23 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg_pred23_2 = nn.Conv3d(num_classes*2, num_classes, 1)

        # self.conv12 = nn.Sequential(nn.Conv3d(2 * n + 4 * n, 2 * n, 3, 1, 1),
        #                             normalization(2 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
        # self.seg_pred12 = nn.Conv3d(2* n, num_classes, 1)
        # self.seg_pred12_2 = nn.Conv3d(num_classes * 2, num_classes, 1)


        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        self.upconv5 = nn.Conv3d(16*n,16*n,1)
        self.upconv4 = nn.Conv3d(16*n,16*n, 1)
        self.upconv3 = nn.Conv3d(8*n, 8*n, 1)
        # self.upconv2 = nn.Conv3d(4 * n, 4 * n, 1)
        self.deeps = [1, 2, 4, 8, 16]
        self.mix_c_f = DMAM(image_size=128,
                            input_channle=n,
                            deeps=self.deeps, )
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
        # print("x4"
        # print("x2",x2.shape)
        # x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)
        # print(x5.shape)
        # il = self.ilc(x5)
        x1,x2,x3,x4,x5=self.mix_c_f([x1,x2,x3,x4,x5])

        pred5 = self.seg_pred5(x5)
        # sl = self.slc(x5, pred)
        sl = self.slc(x5,pred5)    #v18

        y4 = self.convu4(sl, x4)

        sl = F.interpolate(sl,y4.shape[2:],mode='trilinear')
        pred5 = F.interpolate(pred5,y4.shape[2:],mode='trilinear')
        sl = self.upconv5(sl)
        y4 = torch.concat([y4,sl],dim=1)
        y45 = self.conv45(y4)
        pred45 = self.seg_pred45(y45)
        pred45 = self.seg_pred45_2(torch.concat([pred45,pred5],dim=1))
        sl45 = self.slc45(y45,pred45)

        y3 = self.convu3(sl45, x3)

        # y45 =  F.interpolate(y45, scale_factor=2, mode='trilinear', align_corners=False)
        sl45 = F.interpolate(sl45,y3.shape[2:],mode='trilinear')
        pred45 = F.interpolate(pred45, y3.shape[2:], mode='trilinear')
        sl45 = self.upconv4(sl45)
        y3 = self.conv34(torch.concat([y3,sl45],dim=1))
        pred34 = self.seg_pred34(y3)
        pred34 = self.seg_pred34_2(torch.concat([pred34, pred45], dim=1))
        sl34 = self.slc34(y3,pred34)

        y2 = self.convu2(sl34, x2)

        sl34 = F.interpolate(sl34, y2.shape[2:], mode='trilinear')
        pred34 = F.interpolate(pred34, y2.shape[2:], mode='trilinear')
        sl34 = self.upconv3(sl34)
        y2 = self.conv23(torch.concat([y2, sl34], dim=1))
        pred23 = self.seg_pred23(y2)
        pred23 = self.seg_pred23_2(torch.concat([pred23, pred34], dim=1))
        sl23 = self.slc23(y2, pred23)

        y1 = self.convu1(sl23, x1)

        # sl23 = F.interpolate(sl23, y1.shape[2:], mode='trilinear')
        # pred23 = F.interpolate(pred23, y1.shape[2:], mode='trilinear')
        # # sl23 = self.upconv2(sl23)
        # # y1 = self.conv12(torch.concat([y1, sl23], dim=1))
        # pred12 = self.seg_pred12(y1)
        # pred12 = self.seg_pred12_2(torch.concat([pred12, pred23], dim=1))
        # sl12 = self.slc12(y1, pred12)
        out = self.seg1(y1)



        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred23, out.shape[2:],mode='trilinear')
            pred2 = F.interpolate(pred34, out.shape[2:],mode='trilinear')
            pred3= F.interpolate(pred5, out.shape[2:],mode='trilinear')
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out

class Unet_74(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_74, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)
        # ilc_cfg = {
        #     'feats_channels': 16*n,
        #     'transform_channels': 8*n,
        #     'concat_input': True,
        #     'norm_cfg': {'type': 'InstanceNorm3d'},
        #     'act_cfg': {'type': 'ReLU', 'inplace': True},
        #     'align_corners': False,
        # }

        slc_cfg = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = PAM(**slc_cfg)
        slc_cfg45 = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc45 = PAM(**slc_cfg45)
        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc34 = PAM(**slc_cfg34)

        slc_cfg23 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc23 = PAM(**slc_cfg23)

        self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)

        self.conv45 = nn.Sequential(nn.Conv3d(16 * n *2, 16 * n, 3, 1, 1),
                                    normalization(16 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred45 = nn.Conv3d(16 * n, num_classes, 1)
        self.seg_pred45_2 = nn.Conv3d(num_classes * 2, num_classes, 1)


        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+8*n , 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.seg_pred34 = nn.Conv3d(8*n, num_classes, 1)
        self.seg_pred34_2 = nn.Conv3d(num_classes*2, num_classes, 1)
        self.conv23 = nn.Sequential(nn.Conv3d(4 * n + 8 * n, 4 * n, 3, 1, 1),
                                    normalization(4 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred23 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg_pred23_2 = nn.Conv3d(num_classes*2, num_classes, 1)
        # self.conv12 = nn.Sequential(nn.Conv3d(8 * n + 8 * n + 4 * n, 4 * n, 3, 1, 1),
        #                             normalization(8 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        self.upconv5 = nn.Conv3d(16*n,16*n,1)
        self.upconv4 = nn.Conv3d(16*n,16*n, 1)
        self.upconv3 = nn.Conv3d(8*n, 8*n, 1)


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
        # print("x4"
        # print("x2",x2.shape)
        # x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)
        # print(x5.shape)
        # il = self.ilc(x5)
        pred5 = self.seg_pred5(x5)
        # sl = self.slc(x5, pred)
        sl = self.slc(x5,pred5)    #v18
        y4 = self.convu4(sl, x4)
        sl = F.interpolate(sl,y4.shape[2:],mode='trilinear')
        pred5 = F.interpolate(pred5,y4.shape[2:],mode='trilinear')
        sl = self.upconv5(sl)
        y4 = torch.concat([y4,sl],dim=1)

        y45 = self.conv45(y4)
        pred45 = self.seg_pred45(y45)
        pred45 = self.seg_pred45_2(torch.concat([pred45,pred5],dim=1))
        sl45 = self.slc45(y45,pred45)

        y3 = self.convu3(sl45, x3)

        # y45 =  F.interpolate(y45, scale_factor=2, mode='trilinear', align_corners=False)
        sl45 = F.interpolate(sl45,y3.shape[2:],mode='trilinear')
        pred45 = F.interpolate(pred45, y3.shape[2:], mode='trilinear')
        sl45 = self.upconv4(sl45)
        y3 = self.conv34(torch.concat([y3,sl45],dim=1))
        pred34 = self.seg_pred34(y3)
        pred34 = self.seg_pred34_2(torch.concat([pred34, pred45], dim=1))
        sl34 = self.slc34(y3,pred34)

        y2 = self.convu2(sl34, x2)
        sl34 = F.interpolate(sl34, y2.shape[2:], mode='trilinear')
        pred34 = F.interpolate(pred34, y2.shape[2:], mode='trilinear')
        sl34 = self.upconv3(sl34)
        y2 = self.conv23(torch.concat([y2, sl34], dim=1))
        pred23 = self.seg_pred23(y2)
        pred23 = self.seg_pred23_2(torch.concat([pred23, pred34], dim=1))
        sl23 = self.slc23(y2, pred23)

        y1 = self.convu1(sl23, x1)

        # sl34 = F.interpolate(sl34,y1.shape[2:],mode='trilinear')
        # y2 = F.interpolate(y2, y1.shape[2:], mode='trilinear')
        # y1 = self.conv34(torch.concat([y1, y2, sl34], dim=1))
        # pred12 = self.seg_pred12(y1)
        # sl12 = self.slc12(y1, pred12)
        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred23, out.shape[2:],mode='trilinear')
            pred2 = F.interpolate(pred34, out.shape[2:],mode='trilinear')
            pred3= F.interpolate(pred5, out.shape[2:],mode='trilinear')
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out
class Unet_74_swin(nn.Module):
    def __init__(self, c=4, n=32,embed_dim=48, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_74_swin, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision
        self.conv1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm = normalization(n, norm)
        self.relu = nn.LeakyReLU(inplace=True,negative_slope= 0.01)

        self.swin = SwinTransformerSys3D_encoder(
        pretrained=None,
        img_size=(128, 128, 128),
        patch_size=(4, 4, 4),
        in_chans=n,
        num_classes=n,
        embed_dim=embed_dim,
        depths=[2, 2,2, 2,1],
        depths_decoder=[1,2,2,2, 2],
        num_heads=[3,6, 12, 12,12],
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
        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU_sw(16 * n,16*embed_dim, norm, True)
        self.convu3 = ConvU_sw(8 * n,8*embed_dim, norm)
        self.convu2 = ConvU_sw(4 * n,4*embed_dim, norm)
        self.convu1 = ConvU_sw(2 * n,2*embed_dim, norm)

        slc_cfg = {'feats_channels': 16 * embed_dim,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        slc_cfg45 = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        slc_cfg23 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        self.slc = PAM(**slc_cfg)
        self.slc45 = PAM(**slc_cfg45)
        self.slc34 = PAM(**slc_cfg34)
        self.slc23 = PAM(**slc_cfg23)

        self.seg_pred5 = nn.Conv3d(16*embed_dim, num_classes, 1)
        self.seg_pred45 = nn.Conv3d(16 * n, num_classes, 1)
        self.seg_pred45_2 = nn.Conv3d(num_classes * 2, num_classes, 1)
        self.seg_pred34 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg_pred34_2 = nn.Conv3d(num_classes * 2, num_classes, 1)
        self.seg_pred23 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg_pred23_2 = nn.Conv3d(num_classes * 2, num_classes, 1)

        self.conv45 = nn.Sequential(nn.Conv3d(16 * n *2, 16 * n, 3, 1, 1),
                                    normalization(16 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+8*n , 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.conv23 = nn.Sequential(nn.Conv3d(4 * n + 8 * n, 4 * n, 3, 1, 1),
                                    normalization(4 * n, norm),
                                    nn.LeakyReLU(inplace=True))

        self.upconv5 = nn.Conv3d(16*n,16*n,1,1)
        self.upconv4 = nn.Conv3d(16*n,16*n, 1,1)
        self.upconv3 = nn.Conv3d(8*n, 8*n, 1,1)

        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("x",x.shape)
        B,C,D,H,W = x.shape
        x = self.relu(self.norm(self.conv1(x)))
        x1,x2,x3,x4,x5 = self.swin(x)
        # print("x1",x1.shape)
        # # x2 = self.convd2(x1)
        # print("x2",x2.shape)
        # # x3 = self.convd3(x2)
        # print("x3", x3.shape)
        # print("x4",  x4.shape)
        # print(x5.shape, "x5")
        pred5 = self.seg_pred5(x5)
        sl = self.slc(x5,pred5)    #v18
        y4 = self.convu4(sl, x4)
        # print(y4.shape,"y4")
        sl = F.interpolate(sl,y4.shape[2:],mode='trilinear')
        pred5 = F.interpolate(pred5,y4.shape[2:],mode='trilinear')
        sl = self.upconv5(sl)
        y4 = torch.concat([y4,sl],dim=1)

        y45 = self.conv45(y4)
        # print(y4.shape, "y4")
        pred45 = self.seg_pred45(y45)
        pred45 = self.seg_pred45_2(torch.concat([pred45,pred5],dim=1))
        sl45 = self.slc45(y45,pred45)

        y3 = self.convu3(sl45, x3)
        # print(y3.shape, "y3")

        # y45 =  F.interpolate(y45, scale_factor=2, mode='trilinear', align_corners=False)
        sl45 = F.interpolate(sl45,y3.shape[2:],mode='trilinear')
        pred45 = F.interpolate(pred45, y3.shape[2:], mode='trilinear')
        sl45 = self.upconv4(sl45)
        y3 = self.conv34(torch.concat([y3,sl45],dim=1))
        pred34 = self.seg_pred34(y3)
        pred34 = self.seg_pred34_2(torch.concat([pred34, pred45], dim=1))
        sl34 = self.slc34(y3,pred34)

        y2 = self.convu2(sl34, x2)
        # print(y2.shape, "y2")
        sl34 = F.interpolate(sl34, y2.shape[2:], mode='trilinear')
        pred34 = F.interpolate(pred34, y2.shape[2:], mode='trilinear')
        sl34 = self.upconv3(sl34)
        y2 = self.conv23(torch.concat([y2, sl34], dim=1))
        pred23 = self.seg_pred23(y2)
        pred23 = self.seg_pred23_2(torch.concat([pred23, pred34], dim=1))
        sl23 = self.slc23(y2, pred23)

        y1 = self.convu1(sl23, x1)
        # print(y1.shape, "y1")

        # sl34 = F.interpolate(sl34,y1.shape[2:],mode='trilinear')
        # y2 = F.interpolate(y2, y1.shape[2:], mode='trilinear')
        # y1 = self.conv34(torch.concat([y1, y2, sl34], dim=1))
        # pred12 = self.seg_pred12(y1)
        # sl12 = self.slc12(y1, pred12)
        # print(y1.shape)
        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred23, [D,H,W],mode='trilinear')
            pred2 = F.interpolate(pred34, [D,H,W],mode='trilinear')
            pred3= F.interpolate(pred5,  [D,H,W],mode='trilinear')
            # print(pred3.shape,pred2.shape,pred1.shape)
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out
class Unet_74_swin2(nn.Module):
    def __init__(self, c=4, n=32,embed_dim=60, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_74_swin2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision
        self.conv1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm = normalization(n, norm)
        self.relu = nn.LeakyReLU(inplace=True,negative_slope= 0.01)

        self.swin = SwinTransformerSys3D_encoder(
        pretrained=None,
        img_size=(128, 128, 128),
        patch_size=(4, 4, 4),
        in_chans=n,
        num_classes=n,
        embed_dim=embed_dim,
        depths=[2, 2,2, 2],
        depths_decoder=[2,2,2, 2],
        num_heads=[3,6, 12, 24],
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
        # self.nl4 = External_attention(8 * n)
        # self.convu4 = ConvU_sw(16 * n,16*embed_dim, norm, True)
        self.convu3 = ConvU_sw(8 * n,8*embed_dim, norm,True)
        self.convu2 = ConvU_sw(4 * n,4*embed_dim, norm)
        self.convu1 = ConvU_sw(2 * n,2*embed_dim, norm)

        slc_cfg = {'feats_channels': 8 * embed_dim,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        # slc_cfg45 = {'feats_channels': 16 * n,
        #            'out_feats_channels': 16 * n,
        #            'concat_input': True,
        #            'norm_cfg': {'type': 'InstanceNorm3d'},
        #            'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        slc_cfg23 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }
        self.slc = PAM(**slc_cfg)
        # self.slc45 = SemanticLevelContext_3D_45_d3(**slc_cfg45)
        self.slc34 = PAM(**slc_cfg34)
        self.slc23 = PAM(**slc_cfg23)

        self.seg_pred4 = nn.Conv3d(8*embed_dim, num_classes, 1)
        # self.seg_pred45 = nn.Conv3d(16 * n, num_classes, 1)
        # self.seg_pred45_2 = nn.Conv3d(num_classes * 2, num_classes, 1)
        self.seg_pred34 = nn.Conv3d(8 * n, num_classes, 1)
        self.seg_pred34_2 = nn.Conv3d(num_classes * 2, num_classes, 1)
        self.seg_pred23 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg_pred23_2 = nn.Conv3d(num_classes * 2, num_classes, 1)

        # self.conv45 = nn.Sequential(nn.Conv3d(16 * n *2, 16 * n, 3, 1, 1),
        #                             normalization(16 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
        self.conv34 =nn.Sequential(nn.Conv3d(16 * n , 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.conv23 = nn.Sequential(nn.Conv3d(8 * n , 4 * n, 3, 1, 1),
                                    normalization(4 * n, norm),
                                    nn.LeakyReLU(inplace=True))

        # self.upconv5 = nn.Conv3d(16*n,16*n,1,1)
        self.upconv4 = nn.Conv3d(8*n,8*n, 1,1)
        self.upconv3 = nn.Conv3d(8*n, 4*n, 1,1)

        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.fuse = nn.Sequential(nn.Conv3d(embed_dim*8+n*8,n*8,3,1,1),
                                  normalization(n*8,norm),
                                  nn.LeakyReLU(inplace=True))

    def forward(self, x):
        # print("x",x.shape)
        B,C,D,H,W = x.shape
        x = self.relu(self.norm(self.conv1(x)))
        x1,x2,x3,x4 = self.swin(x)
        # print("x1",x1.shape)
        # # x2 = self.convd2(x1)
        # print("x2",x2.shape)
        # # x3 = self.convd3(x2)
        # print("x3", x3.shape)
        # print("x4",  x4.shape)
        # print(x5.shape, "x5")

        pred4 = self.seg_pred4(x4)
        sl = self.slc(x4,pred4)    #v18

        sl4 = self.fuse(torch.concat([sl,x4],dim=1))

        # y4 = self.convu4(sl, x4)
        # print(y4.shape,"y4")
        # sl = F.interpolate(sl,y3.shape[2:],mode='trilinear')
        # pred4 = F.interpolate(pred4,y3.shape[2:],mode='trilinear')
        # sl = self.upconv5(sl)
        # y4 = torch.concat([y4,sl],dim=1)
        #
        # y45 = self.conv45(y4)
        # print(y4.shape, "y4")
        # pred45 = self.seg_pred45(y45)
        # pred45 = self.seg_pred45_2(torch.concat([pred45,pred5],dim=1))
        # sl45 = self.slc45(y45,pred45)

        y3 = self.convu3(sl4, x3)
        # print(y3.shape, "y3")

        # y45 =  F.interpolate(y45, scale_factor=2, mode='trilinear', align_corners=False)
        sl45 = F.interpolate(sl4,y3.shape[2:],mode='trilinear')
        pred45 = F.interpolate(pred4, y3.shape[2:], mode='trilinear')
        sl45 = self.upconv4(sl45)
        y3 = self.conv34(torch.concat([y3,sl45],dim=1))
        pred34 = self.seg_pred34(y3)
        pred34 = self.seg_pred34_2(torch.concat([pred34, pred45], dim=1))
        sl34 = self.slc34(y3,pred34)

        y2 = self.convu2(sl34, x2)
        # print(y2.shape, "y2")
        sl34 = F.interpolate(sl34, y2.shape[2:], mode='trilinear')
        pred34 = F.interpolate(pred34, y2.shape[2:], mode='trilinear')
        sl34 = self.upconv3(sl34)
        y2 = self.conv23(torch.concat([y2, sl34], dim=1))
        pred23 = self.seg_pred23(y2)
        pred23 = self.seg_pred23_2(torch.concat([pred23, pred34], dim=1))
        sl23 = self.slc23(y2, pred23)

        y1 = self.convu1(sl23, x1)
        # print(y1.shape, "y1")

        # sl34 = F.interpolate(sl34,y1.shape[2:],mode='trilinear')
        # y2 = F.interpolate(y2, y1.shape[2:], mode='trilinear')
        # y1 = self.conv34(torch.concat([y1, y2, sl34], dim=1))
        # pred12 = self.seg_pred12(y1)
        # sl12 = self.slc12(y1, pred12)
        # print(y1.shape)
        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred23, [D,H,W],mode='trilinear')
            pred2 = F.interpolate(pred34, [D,H,W],mode='trilinear')
            pred3= F.interpolate(pred4,  [D,H,W],mode='trilinear')
            # print(pred3.shape,pred2.shape,pred1.shape)
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out
class Unet_81(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_81, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)
        # ilc_cfg = {
        #     'feats_channels': 16*n,
        #     'transform_channels': 8*n,
        #     'concat_input': True,
        #     'norm_cfg': {'type': 'InstanceNorm3d'},
        #     'act_cfg': {'type': 'ReLU', 'inplace': True},
        #     'align_corners': False,
        # }

        slc_cfg = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = PAM(**slc_cfg)
        slc_cfg45 = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc45 = PAM(**slc_cfg45)
        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc34 = PAM(**slc_cfg34)

        slc_cfg23 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc23 = PAM(**slc_cfg23)

        self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)

        self.conv45 = nn.Sequential(nn.Conv3d(16 * n *2, 16 * n, 3, 1, 1),
                                    normalization(16 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred45 = nn.Conv3d(16 * n, num_classes, 1)
        self.seg_pred45_2 = nn.Sequential(nn.Conv3d(num_classes, num_classes, 1),
                                   nn.InstanceNorm3d(3),
                                   nn.LeakyReLU(inplace=True))
        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+8*n , 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.seg_pred34 = nn.Conv3d(8*n, num_classes, 1)
        self.seg_pred34_2 = nn.Sequential(nn.Conv3d(num_classes, num_classes,  1),
                                          nn.InstanceNorm3d(3),
                                    nn.LeakyReLU(inplace=True))
        self.conv23 = nn.Sequential(nn.Conv3d(4 * n + 8 * n, 4 * n, 3, 1, 1),
                                    normalization(4 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred23 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg_pred23_2 = nn.Sequential(nn.Conv3d(num_classes, num_classes, 1,),
                                          nn.InstanceNorm3d(3),
                                          nn.LeakyReLU(inplace=True))
        # self.conv12 = nn.Sequential(nn.Conv3d(8 * n + 8 * n + 4 * n, 4 * n, 3, 1, 1),
        #                             normalization(8 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        self.upconv5 = nn.Conv3d(16*n,16*n,1)
        self.upconv4 = nn.Conv3d(16*n,16*n, 1)
        self.upconv3 = nn.Conv3d(8*n, 8*n, 1)


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
        # print("x4"
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)
        # print(x5.shape)
        # il = self.ilc(x5)
        pred5 = self.seg_pred5(x5)
        # sl = self.slc(x5, pred)
        sl = self.slc(x5,pred5)    #v18
        y4 = self.convu4(sl, x4)
        sl = F.interpolate(sl,y4.shape[2:],mode='trilinear')
        pred5 = F.interpolate(pred5,y4.shape[2:],mode='trilinear')
        sl = self.upconv5(sl)
        y4 = torch.concat([y4,sl],dim=1)

        y45 = self.conv45(y4)
        pred45 = self.seg_pred45(y45)
        pred5 = self.seg_pred45_2(pred5)
        pred45 = pred45+pred5
        sl45 = self.slc45(y45,pred45)

        y3 = self.convu3(sl45, x3)

        # y45 =  F.interpolate(y45, scale_factor=2, mode='trilinear', align_corners=False)
        sl45 = F.interpolate(sl45,y3.shape[2:],mode='trilinear')
        pred45 = F.interpolate(pred45, y3.shape[2:], mode='trilinear')
        sl45 = self.upconv4(sl45)
        y3 = self.conv34(torch.concat([y3,sl45],dim=1))
        pred34 = self.seg_pred34(y3)
        pred45 = self.seg_pred34_2(pred45)
        pred34 = pred34+ pred45
        sl34 = self.slc34(y3,pred34)

        y2 = self.convu2(sl34, x2)
        sl34 = F.interpolate(sl34, y2.shape[2:], mode='trilinear')
        pred34 = F.interpolate(pred34, y2.shape[2:], mode='trilinear')
        sl34 = self.upconv3(sl34)
        y2 = self.conv23(torch.concat([y2, sl34], dim=1))
        pred23 = self.seg_pred23(y2)
        pred34 = self.seg_pred23_2(pred34)
        pred23 = pred23+pred34
        sl23 = self.slc23(y2, pred23)


        y1 = self.convu1(sl23, x1)

        # sl34 = F.interpolate(sl34,y1.shape[2:],mode='trilinear')
        # y2 = F.interpolate(y2, y1.shape[2:], mode='trilinear')
        # y1 = self.conv34(torch.concat([y1, y2, sl34], dim=1))
        # pred12 = self.seg_pred12(y1)
        # sl12 = self.slc12(y1, pred12)
        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred23, out.shape[2:],mode='trilinear')
            pred2 = F.interpolate(pred34, out.shape[2:],mode='trilinear')
            pred3= F.interpolate(pred5, out.shape[2:],mode='trilinear')
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out

class Unet_swinunter(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=4,deep_supervision=False):
        super(Unet_swinunter, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.deep_supervision=deep_supervision

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        # self.nl4 = External_attention(8 * n)
        self.convu4 = ConvU(16 * n, norm, True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)
        # ilc_cfg = {
        #     'feats_channels': 16*n,
        #     'transform_channels': 8*n,
        #     'concat_input': True,
        #     'norm_cfg': {'type': 'InstanceNorm3d'},
        #     'act_cfg': {'type': 'ReLU', 'inplace': True},
        #     'align_corners': False,
        # }

        slc_cfg = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc = PAM(**slc_cfg)
        slc_cfg45 = {'feats_channels': 16 * n,
                   'out_feats_channels': 16 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc45 = PAM(**slc_cfg45)
        slc_cfg34 = {'feats_channels': 8 * n,
                   'out_feats_channels': 8 * n,
                   'concat_input': True,
                   'norm_cfg': {'type': 'InstanceNorm3d'},
                   # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                   'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc34 = PAM(**slc_cfg34)

        slc_cfg23 = {'feats_channels': 4*n,
                     'out_feats_channels': 4* n,
                     'concat_input': True,
                     'norm_cfg': {'type': 'InstanceNorm3d'},
                     # 'norm_cfg1d':{'type': 'InstanceNorm1d'},
                     'act_cfg': {'type': 'LeakyReLU', 'inplace': True}, }

        # self.ilc = ImageLevelContext_3D(**ilc_cfg)
        self.slc23 = PAM(**slc_cfg23)

        self.seg_pred5 = nn.Conv3d(16 * n, num_classes, 1)

        self.conv45 = nn.Sequential(nn.Conv3d(16 * n *2, 16 * n, 3, 1, 1),
                                    normalization(16 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred45 = nn.Conv3d(16 * n, num_classes, 1)
        self.seg_pred45_2 = nn.Conv3d(num_classes * 2, num_classes, 1)


        self.conv34 =nn.Sequential(nn.Conv3d(16 * n+8*n , 8*n, 3,1,1),
                                   normalization(8*n, norm),
                                   nn.LeakyReLU(inplace=True))
        self.seg_pred34 = nn.Conv3d(8*n, num_classes, 1)
        self.seg_pred34_2 = nn.Conv3d(num_classes*2, num_classes, 1)
        self.conv23 = nn.Sequential(nn.Conv3d(4 * n + 8 * n, 4 * n, 3, 1, 1),
                                    normalization(4 * n, norm),
                                    nn.LeakyReLU(inplace=True))
        self.seg_pred23 = nn.Conv3d(4 * n, num_classes, 1)
        self.seg_pred23_2 = nn.Conv3d(num_classes*2, num_classes, 1)
        # self.conv12 = nn.Sequential(nn.Conv3d(8 * n + 8 * n + 4 * n, 4 * n, 3, 1, 1),
        #                             normalization(8 * n, norm),
        #                             nn.LeakyReLU(inplace=True))
        self.seg1 = nn.Conv3d(2 * n, num_classes, 1)

        self.upconv5 = nn.Conv3d(16*n,16*n,1)
        self.upconv4 = nn.Conv3d(16*n,16*n, 1)
        self.upconv3 = nn.Conv3d(8*n, 8*n, 1)


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
        # print("x4"
        # print("x2",x2.shape)
        x3 = self.convd3(x2)
        # print("x3", x3.shape), x4.shape)
        x5 = self.convd5(x4)


        # print(x5.shape)
        # il = self.ilc(x5)
        pred5 = self.seg_pred5(x5)
        # sl = self.slc(x5, pred)
        sl = self.slc(x5,pred5)    #v18
        y4 = self.convu4(sl, x4)
        sl = F.interpolate(sl,y4.shape[2:],mode='trilinear')
        pred5 = F.interpolate(pred5,y4.shape[2:],mode='trilinear')
        sl = self.upconv5(sl)
        y4 = torch.concat([y4,sl],dim=1)

        y45 = self.conv45(y4)
        pred45 = self.seg_pred45(y45)
        pred45 = self.seg_pred45_2(torch.concat([pred45,pred5],dim=1))
        sl45 = self.slc45(y45,pred45)

        y3 = self.convu3(sl45, x3)

        # y45 =  F.interpolate(y45, scale_factor=2, mode='trilinear', align_corners=False)
        sl45 = F.interpolate(sl45,y3.shape[2:],mode='trilinear')
        pred45 = F.interpolate(pred45, y3.shape[2:], mode='trilinear')
        sl45 = self.upconv4(sl45)
        y3 = self.conv34(torch.concat([y3,sl45],dim=1))
        pred34 = self.seg_pred34(y3)
        pred34 = self.seg_pred34_2(torch.concat([pred34, pred45], dim=1))
        sl34 = self.slc34(y3,pred34)

        y2 = self.convu2(sl34, x2)

        sl34 = F.interpolate(sl34, y2.shape[2:], mode='trilinear')
        pred34 = F.interpolate(pred34, y2.shape[2:], mode='trilinear')
        sl34 = self.upconv3(sl34)
        y2 = self.conv23(torch.concat([y2, sl34], dim=1))
        pred23 = self.seg_pred23(y2)
        pred23 = self.seg_pred23_2(torch.concat([pred23, pred34], dim=1))
        sl23 = self.slc23(y2, pred23)

        y1 = self.convu1(sl23, x1)

        # sl34 = F.interpolate(sl34,y1.shape[2:],mode='trilinear')
        # y2 = F.interpolate(y2, y1.shape[2:], mode='trilinear')
        # y1 = self.conv34(torch.concat([y1, y2, sl34], dim=1))
        # pred12 = self.seg_pred12(y1)
        # sl12 = self.slc12(y1, pred12)
        out = self.seg1(y1)


        if self.training and self.deep_supervision:
            out_all = [out]
            pred1 = F.interpolate(pred23, out.shape[2:],mode='trilinear')
            pred2 = F.interpolate(pred34, out.shape[2:],mode='trilinear')
            pred3= F.interpolate(pred5, out.shape[2:],mode='trilinear')
            # pred4 = F.interpolate(pred4, out.shape[2:])
            out_all.append(pred1)
            out_all.append(pred2)
            out_all.append(pred3)
            # out_all.append(pred4)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        return out
if __name__ == "__main__":
    import torch as t
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    print('-----' * 5)

    a = t.randn(2,5,128,128,128)
    deeps = [16,32,64,128,256]
    net = Unet_74_mc(c=5, n=32, dropout=0.5, norm='gn',  num_classes=3,deep_supervision=True)
    # net = Unetd4(c=5,  dropout=0.5, norm='gn', deeps=deeps,num_classes=3,deep_supervision=True)
    out = net(a)
    # print(net)
    print(out.shape)
    flops =FlopCountAnalysis(net,a)
    print("flops:",flops.total())
    print(parameter_count_table(net))
