import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random

from moudels.backbones.you import Unet
from moudels.backbones.unet_swin import *
from torch.nn.functional import interpolate

class cascade_all_4(nn.Module):
    def __init__(self, c=4, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None):
        super(cascade_all_4, self).__init__()
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, )
        self.stage2 = Unet(c=c+num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)
        # self.stage1.load_from(pretrain_path)
        # self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)
        # print(x.shape,x1.shape,x2.shape)
        return x2

class cascade_all_6(nn.Module):
    def __init__(self, c=4, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None):
        super(cascade_all_6, self).__init__()
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, )
        # self.stage2 = Unet(c=c+num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage2 = Swin_Unet_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,
                                  pretrain=pretrain_path)

        # self.stage1.load_from(pretrain_path)
        self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)
        # print(x.shape,x1.shape,x2.shape)
        return x2
class cascade_all_6_od(nn.Module):
    def __init__(self, c=4, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None):
        super(cascade_all_6_od, self).__init__()
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, )
        # self.stage2 = Unet(c=c+num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage2 = Swin_Unet_od_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,
                                  pretrain=pretrain_path)

        # self.stage1.load_from(pretrain_path)
        self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)
        # print(x.shape,x1.shape,x2.shape)
        return x2
class cascade_all_7(nn.Module):
    def __init__(self, c=4, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None):
        super(cascade_all_7, self).__init__()
        self.stage1 = Swin_Unet_10(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, pretrain=pretrain_path)
        # self.stage2 = Unet(c=c+num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage2 = Swin_Unet_10(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,
                                  pretrain=pretrain_path)

        self.stage1.load_from(pretrain_path)
        self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)
        # print(x.shape,x1.shape,x2.shape)
        return x2
class cascade_all_8(nn.Module):
    def __init__(self, c=4, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None):
        super(cascade_all_8, self).__init__()
        self.stage1 = Swin_Unet_OD_10(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, pretrain=pretrain_path)
        # self.stage2 = Unet(c=c+num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage2 = Swin_Unet_OD_10(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,
                                  pretrain=pretrain_path)

        self.stage1.load_from(pretrain_path)
        self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)
        # print(x.shape,x1.shape,x2.shape)
        return x2
class cascade_all_9(nn.Module):
    def __init__(self, c=4, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None):
        super(cascade_all_9, self).__init__()
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        # self.stage2 = Unet(c=c+num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage2 = Swin_Unet_od_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,
                                  pretrain=pretrain_path)

        # self.stage1.load_from(pretrain_path)
        # self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)
        # print(x.shape,x1.shape,x2.shape)
        return x2
class cascade_all_10(nn.Module):
    def __init__(self, c=5, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,pretrain_path=None):
        super(cascade_all_10, self).__init__()
        # self.stage1 = Swin_Unet_OD_10_1(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage2 = Swin_Unet_OD_10_1(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,
                                  pretrain=pretrain_path)

        # self.stage1.load_from(pretrain_path)
        # self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)
        # print(x.shape,x1.shape,x2.shape)
        return x2
class cascade_all_11(nn.Module):
    def __init__(self, c=5, n1=16,n2=32, dropout=0.5, norm='gn',deep_supervision=False, num_classes=3,pretrain_path=None):
        super(cascade_all_11, self).__init__()
        self.deep_supervision=deep_supervision
        # self.stage1 = Swin_Unet_OD_10_1(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        # self.stage2 = Unet(c=c+num_classes, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, )  #v36
        # self.stage2 = Unet_od(c=c + num_classes, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, )  #v47
        self.stage2 = Swin_Unet_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)  #v54-0
        # self.stage2 = Swin_Unet_od_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes, ) #v38-v40  v48
        # self.stage2 = Swin_Unet_9_b(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,) #v34 v35
        # self.stage2 = Swin_Unet_dcd_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes, )  #v42 v44
        # self.stage2 = Unet_dcd(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes, )  #v43weipaowan
        # self.stage1.load_from(pretrain_path)
        # self.stage2.load_from(pretrain_path)

    def forward(self, x):
        # print(x.shape)
        x1 = self.stage1(x)
        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)

        # x2 ,up= self.stage2(x2)#35
        if self.training and self.deep_supervision:
            out_all = [x2]

            out_all.append(x1)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        # print(x.shape,x1.shape,x2.shape)

        return x2

class cascade_all_three(nn.Module):

    def __init__(self, c=5, n1=16,n2=32, dropout=0.5, norm='gn',deep_supervision=False, num_classes=3,pretrain_path=None):
        super(cascade_all_three, self).__init__()
        self.deep_supervision=deep_supervision
        self.stage2_size = 96
        # self.stage1 = Swin_Unet_OD_10_1(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage1 = Unet(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        # self.stage2 = Unet(c=c+num_classes, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, )  #v36
        # self.stage2 = Unet_od(c=c + num_classes, n=n1, dropout=0.5, norm='gn', num_classes=num_classes, )  #v47
        self.stage2 = Swin_Unet_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,)  #v54-0
        # self.stage2 = Swin_Unet_od_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes, ) #v38-v40  v48
        # self.stage2 = Swin_Unet_9_b(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes,) #v34 v35
        # self.stage2 = Swin_Unet_dcd_9(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes, )  #v42 v44
        # self.stage2 = Unet_dcd(c=c + num_classes, n=n2, dropout=0.5, norm='gn', num_classes=num_classes, )  #v43weipaowan
        # self.stage1.load_from(pretrain_path)
        # self.stage2.load_from(pretrain_path)

    def forward(self, x):
        # print(x.shape)
        x1 = self.stage1(x)
        batch_idxs,num_class,D,H,W = x1.size()
        stage1 = (torch.sigmoid(x1) > 0.5).int()
        for batch_idx in range(batch_idxs):
            st = stage1[batch_idx][1]
            # print(st)
            if st.sum()==0: continue
            # print(st.shape)
            pred1_wt_idx = list(torch.where(st>0))
            print(len(pred1_wt_idx),len(pred1_wt_idx[0]),type(pred1_wt_idx))
            x_idx = pred1_wt_idx[0]
            y_idx = pred1_wt_idx[1]
            z_idx = pred1_wt_idx[2]
            print(len(x_idx),len(y_idx),len(z_idx))
            random_muber = torch.randint(0,len(x_idx),[1])
            # print(random_muber.shape)
            print(x_idx[random_muber],y_idx[random_muber],z_idx[random_muber])

            # if random_muber<self.stage2_size/2:
            #     ne = st[0:96,0:96,0:96]
            # elif random_muber>H-self.stage2_size/2-1:
            #     ne = st[H-self.stage2_size-1:H-1,31:127,31:127]
            # else:
            #     ne = st[random_muber-48:random_muber+48,random_muber-48:random_muber+48,random_muber-48:random_muber+48]
            # print(ne.shape)

            # else: ne = st[]
            # print(ne.shape)


            # max_x,min_x = torch.argmax(pred1_wt_idx,1)
            # max_y, min_y =torch.argmax(pred1_wt_idx,dim=2)
            # max_z,min_z = torch.argmax(pred1_wt_idx,dim=3)
            # print(max_x,min_x,max_y,min_y,max_z,min_z)

        x2 = torch.cat([x, x1], 1)
        x2 = self.stage2(x2)

        # x2 ,up= self.stage2(x2)#35
        if self.training and self.deep_supervision:
            out_all = [x2]

            out_all.append(x1)
            # out_all.append(up)# 35
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        # print(x.shape,x1.shape,x2.shape)

        return x2
class cascade_swin_od_deep(nn.Module):
    def __init__(self, c=5, n1=16,n2=32, dropout=0.5, norm='gn', num_classes=3,
                 deep_supervision=False,
                 pretrain_path=None):
        super(cascade_swin_od_deep, self).__init__()
        self.deep_supervision = deep_supervision
        # self.stage1 = Swin_Unet_OD_10_1(c=c, n=n1, dropout=0.5, norm='gn', num_classes=num_classes,)
        self.stage1 = Unet(c=c, n=n1, dropout=dropout, norm=norm, num_classes=num_classes,)

        # self.stage2 = Unet(c=c + num_classes, n=n2, dropout=dropout, norm=norm, num_classes=num_classes, ) #v23
        # self.stage2 = Swin_Unet_od_8(c=c + num_classes, n=n2, dropout=dropout, norm=norm, num_classes=num_classes,) #v22
        # self.stage2 = Swin_Unet_8(c=c + num_classes, n=n2, dropout=dropout, norm=norm, num_classes=num_classes, ) #v24
        # self.stage2 = Unet_od(c=c + num_classes, n=n2, dropout=dropout, norm=norm, num_classes=num_classes,) #v25
        # self.stage2 = Swin_Unet_od_9(c=c + num_classes, n=n2, dropout=dropout, norm=norm, num_classes=num_classes, ) #weipao
        # self.stage2 = Swin_Unet_od_3(c=c + num_classes, n=n2, dropout=dropout, norm=norm,
        #                              num_classes=num_classes, )  # v27 v28
        self.stage2 =Swin_Unet_od_9(c=c + num_classes, n=n2, dropout=dropout, norm=norm,
                                     num_classes=num_classes, )  # v30



        # self.stage1.load_from(pretrain_path)
        # self.stage2.load_from(pretrain_path)

    def forward(self, x):
        x1 = self.stage1(x)
        out = torch.cat([x, x1], 1)
        out = self.stage2(out)
        if self.training and self.deep_supervision:
            out_all = [out]

            out_all.append(x1)
            # print(torch.stack(out_all, dim=1).shape)
            return torch.stack(out_all, dim=1)
        # print(x.shape,x1.shape,x2.shape)
        return out
if __name__ == "__main__":
    import torch as t
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    print('-----' * 5)
    # rga = t.randn(2, 4, 128, 128, 128)
    # rgb = t.randn(1, 3, 352, 480, 150)
    # net = Unet()
    # out = net(rga)
    # print(out.shape)
    a = t.randn(2,5,128,128,128)
    pretrain_path = "/home/user/4TB/Chenbonian/medic-segmention/pratrain/swin_tiny_patch4_window7_224_22k.pth"
    # dac = torch.load(path)
    # print(dac['model'])
    net = cascade_all_11(c=5, n1=16,n2=16, dropout=0.5, norm='gn', num_classes=3,pretrain_path=pretrain_path)
    # net.load_state_dict(dac['state_dict'])
    out = net(a)
    # print(dac['state_dict'].keys())
    # kkk=dac['state_dict']
    # print(kkk['model'].keys())
    print(out.shape)
    print("paramr", parameter_count_table(net))
