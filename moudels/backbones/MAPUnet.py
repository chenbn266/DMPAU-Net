from moudels.backbones.swintransformer import SwinTransformerSys3D
import copy
from moudels.backbones.ODconv3D import ODConv3d
from moudels.backbones.Dynamic_conv import Dynamic_conv3d
from moudels.backbones.dcd import conv_dy
from torch.nn.functional import interpolate
from moudels.backbones.CMMM import *
from moudels.backbones.you import ConvU, ConvD
from moudels.isnet.imagelevel import *
from moudels.backbones.CMMM import *

#####整个解构
class MAPUnet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=4,
                 deep_supervision=False,
                 use_bottom_slc=False,
                 use_top_slc=False,
                 use_multiple_semantics=False,
                 use_mix=False,
                 use_prototype=True,
                 groups = [3,5,7,11],
                 up_size = 3,
                 ):
        super(MAPUnet, self).__init__()

        self.deeps = [1, 2, 4, 8]
        self.use_bottom_slc = use_bottom_slc
        self.use_mix = use_mix
        self.use_multiple_semantics = use_multiple_semantics
        self.use_top_slc = use_top_slc
        self.use_prototype = use_prototype
        self.deep_supervision = deep_supervision
        self.groups = groups

        self.convd1 = ConvD(c, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, dropout, norm)
        self.convd3 = ConvD(2 * n, 4 * n, dropout, norm)
        self.convd4 = ConvD(4 * n, 8 * n, dropout, norm)
        self.convd5 = ConvD(8 * n, 16 * n, dropout, norm)

        self.seg_bottleneck = nn.Conv3d(n * 16, num_classes, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.decoder = nn.ModuleList()
        self.seg = nn.ModuleList()
        self.prototype = nn.ModuleList()
        self.multiple_semantics = nn.ModuleList()

        for i in range(4):
            input_channle = n * 2 ** (i + 1)
            decoder = ConvU(input_channle, norm)
            seg = nn.Conv3d(input_channle, num_classes, 1)
            if self.use_prototype:
                if i == 0 and self.use_top_slc == False:
                    slc = None
                else:
                    pam_cfg = {'feats_channels': input_channle,
                               'out_feats_channels': input_channle,
                               'concat_input': True,
                               'norm_cfg': {'type': 'GroupNorm'},
                               'act_cfg': {'type': 'GELU'},
                               }
                    slc = PAM(**pam_cfg)
                self.prototype.append(slc)

            if i == 3:
                decoder = ConvU(n * 2 ** (i + 1), norm, True)
            self.decoder.append(decoder)
            self.seg.append(seg)

            if use_multiple_semantics:
                ms = nn.Sequential(nn.Conv3d(num_classes * 2, num_classes, 1),
                                   )

                self.multiple_semantics.append(ms)

        if self.use_bottom_slc:
            slc_cfg5 = {'feats_channels': 16 * n,
                        'out_feats_channels': 16 * n,
                        'concat_input': True,
                        'norm_cfg': {'type': 'GroupNorm'},
                        'act_cfg': {'type': 'GELU'},
                        }
            self.slc5 = PAM(**slc_cfg5)

        if self.use_mix:
            # self.mix_c_f = mix_channle_fuse2(image_size=128,
            #                                  input_channle=n,
            #                                  deeps=[1, 2, 4, 8],
            #                                  group=self.groups,
            #                                  kernel_size=up_size,
            #                                  fuse_to_size=32,
            #                                  mid_channle=32)
            self.mix_c_f = DMAM(image_size=128,
                                input_channle=n,
                                deeps=[1, 2, 4, 8],
                                group=self.groups,
                                )

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
        if self.use_mix == True:
            x_mix = self.mix_c_f([x1, x2, x3, x4])
            x_mix.append(x5)
            # x_mix = [x1, x2, x3, x4, x5]
        else:
            x_mix = [x1, x2, x3, x4, x5]


        seg5 = self.seg_bottleneck(x_mix[-1])
        if self.use_bottom_slc:
            x_mix[-1] = self.slc5(x_mix[-1], seg5)
        y = []
        out = [seg5]
        for i in range(4):
            if i == 0:
                y1 = self.decoder[-(i + 1)](x_mix[-1], x_mix[-2])
            else:
                y1 = self.decoder[-(i + 1)](y[-1], x_mix[-(i + 2)])

            pred = self.seg[-(i + 1)](y1)


            if self.use_prototype:
                if i == 3 and self.use_top_slc == False:
                    y1 = y1
                else:
                    y1 = self.prototype[-(i + 1)](y1, pred)
            y.append(y1)
            if self.multiple_semantics:
                y1_pred = self.seg[-(i + 1)](y1)
                pred = pred+y1_pred
            out.append(pred)

        output = out[-1]
        if self.training and self.deep_supervision:
            out_all = [output]
            for i in range(4):
                j = -(i + 2)
                out[j] = F.interpolate(out[j], output.shape[2:], mode='trilinear')
                out_all.append(out[j])

            return torch.stack(out_all, dim=1)
        return output


if __name__ == "__main__":
    import torch as t
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    print('-----' * 5)
    rga = t.randn(2, 2, 128, 128, 128)
    net = MAPUnet(c=2, n=32, dropout=0.5, norm='gn', num_classes=3, deep_supervision=False,
                  use_prototype=False, use_mix=True, use_bottom_slc=True, use_multiple_semantics=False, use_top_slc=False)

    print("paramr", parameter_count_table(net))
    out = net(rga)
    print(out[0].shape, out[1].shape)
