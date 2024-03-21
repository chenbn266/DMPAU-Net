import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from moudels.isnet.imagelevel import *

from fvcore.nn import FlopCountAnalysis, parameter_count_table

class acnet(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(acnet, self).__init__()
        self.gn = nn.GroupNorm(num_channels=inplanes,num_groups=4)

        self.conv1 = nn.Conv3d(in_channels=inplanes*2,out_channels=planes,kernel_size=3,padding=1,stride=1)

        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        # gn_x=self.gn(x)
        # w_gamma = self.gn.weight/sum(self.gn.weight)
        # reweigts = torch.sigmoid(w_gamma)
        # gate_treshold = self.pool(x)
        # print(gate_treshold.shape,reweigts.shape)
        #
        # mask = reweigts

        pool = self.pool(x)
        pool = F.interpolate(pool,scale_factor=x.size()[2:],mode='trilinear')
        sub = x-pool
        weights = torch.sigmoid(sub)
        x = x*weights

        return x
class SemanticLevelContext_3D_12(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False, norm_cfg=None, act_cfg=None):
        super(SemanticLevelContext_3D_12, self).__init__()

        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv3d(feats_channels*2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.norm = BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg)



    '''forward'''
    def forward(self, x, preds):
        inputs = x
        batch_size, num_channels, h, w,d = x.size()
        num_classes = preds.size(1)   #psp ispp

        c_p = self.avgpool(preds)
        c_p = c_p.reshape(batch_size,num_classes,-1)
        c_p_weight = F.softmax(c_p, dim=1)

        feats_sl = torch.zeros(batch_size, h*w*d, num_channels).type_as(x)
        for batch_idx in range(batch_size):
            # (C, H, W，d), (num_classes, H, W，d) --> (H*W, C), (H*W, num_classes)
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)

            # (H*W, )
            argmax = preds_iter.argmax(1)
            # argmax_s = preds_iter.argmax(0)
            for clsid in range(num_classes):
                mask = (argmax == clsid)
                if mask.sum() == 0: continue
                feats_iter_cls = feats_iter[mask]
                preds_iter_cls = preds_iter[:, clsid][mask]
                # mean= torch.mean(preds_iter_cls)
                # print(mean)
                weight = F.softmax(preds_iter_cls, dim=0)
                # feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1)
                # feats_iter_cls = (feats_iter_cls +mean)* weight.unsqueeze(-1)  #更高级的融合方式  SEnet，一个空间一个通道
                k = c_p_weight[batch_idx,clsid,:]
                feats_iter_cls = feats_iter_cls*weight.unsqueeze(-1)
                feats_iter_cls = feats_iter_cls.sum(0)    #
                feats_sl[batch_idx][mask] = feats_iter_cls
        feats_sl = feats_sl.reshape(batch_size, h, w,d, num_channels)
        feats_sl = feats_sl.permute(0, 4, 1, 2,3).contiguous()
        # feats_sl = self.correlate_net(inputs, feats_sl)  #
        feats_sl =self.norm(inputs*feats_sl)

        feats_sl = self.correlate_net(feats_sl,feats_sl)


        # if hasattr(self, 'bottleneck'):
        #     feats_sl = self.bottleneck(torch.cat([x, feats_sl], dim=1)) #cross attion
            # feats_sl = self.correlate_net(x, feats_sl)
            # feats_sl = self.bottleneck(feats_sl)
        return feats_sl

# class SRU
if __name__ == "__main__":
    import torch as t

    print('-----' * 5)
    # rga = t.randn(2, 4, 128, 128, 128)
    # rgb = t.randn(1, 3, 352, 480, 150)
    # net = Unet()
    # out = net(rga)
    # print(out.shape)
    a = t.randn(2,32,128,128,128)
    # a = t.randn(2,512,8,8,8)
    deeps = [16,32,64,128,256]
    net = acnet(inplanes=a.shape[1], planes=a.shape[1],kernel_size=3,stride=1,padding=1)
    # net = Unetd4(c=5,  dropout=0.5, norm='gn', deeps=deeps,num_classes=3,deep_supervision=True)
    out,out2 = net(a)
    # print(net)
    print(out.shape,out2.shape)
    # flops =FlopCountAnalysis(net,a)
    #
    # print("flops:",flops.total())
    # print(parameter_count_table(net))