'''
Function:
    Implementation of ImageLevelContext
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from moudels.backbones.Transformer import *
class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)
    '''forward'''
    def forward(self, x):
        return x * self.act(x + 3) / 6
class HardSigmoid(nn.Module):
    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HardSigmoid, self).__init__()
        assert divisor != 0, 'divisor is not allowed to be equal to zero'
        self.bias = bias
        self.divisor = divisor
        self.min_value = min_value
        self.max_value = max_value
    '''forward'''
    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, **kwargs):
        super(LayerNorm2d, self).__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]
    '''forward'''
    def forward(self, x):
        assert x.dim() == 4, f'LayerNorm2d only supports inputs with shape (N, C, H, W), but got tensor with shape {x.shape}'
        x = F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)
        return x
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query,
                 query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm,
                 value_out_norm, matmul_norm, with_out_project, norm_cfg=None, act_cfg=None):
        super(SelfAttentionBlock, self).__init__()
        # key project
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
            use_norm=key_query_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # query project
        if share_key_query:
            assert key_in_channels == query_in_channels
            self.query_project = self.key_project
        else:
            self.query_project = self.buildproject(
                in_channels=query_in_channels,
                out_channels=transform_channels,
                num_convs=key_query_num_convs,
                use_norm=key_query_norm,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        # value project
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels if with_out_project else out_channels,
            num_convs=value_out_num_convs,
            use_norm=value_out_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # out project
        self.out_project = None
        if with_out_project:
            self.out_project = self.buildproject(
                in_channels=transform_channels,
                out_channels=out_channels,
                num_convs=value_out_num_convs,
                use_norm=value_out_norm,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        # downsample
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.transform_channels = transform_channels
    '''forward'''
    def forward(self, query_feats, key_feats):
        batch_size = query_feats.size(0)
        # print(type(query_feats))
        query = self.query_project(query_feats)
        if self.query_downsample is not None: query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()
        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context
    '''build project'''
    def buildproject(self, in_channels, out_channels, num_convs, use_norm, norm_cfg, act_cfg):
        if use_norm:
            convs = [nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )]
            for _ in range(num_convs - 1):
                convs.append(nn.Sequential(
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                ))
        else:
            convs = [nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]

def BuildActivation(act_cfg):
    if act_cfg is None: return nn.Identity()
    act_cfg = copy.deepcopy(act_cfg)
    # supported activations
    supported_activations = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'ReLU6': nn.ReLU6,
        'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid,
        'HardSwish': HardSwish,
        'LeakyReLU': nn.LeakyReLU,
        'HardSigmoid': HardSigmoid,
    }
    # build activation
    act_type = act_cfg.pop('type')
    activation = supported_activations[act_type](**act_cfg)
    # return
    return activation
def BuildNormalization(placeholder, norm_cfg):
    if norm_cfg is None: return nn.Identity()
    norm_cfg = copy.deepcopy(norm_cfg)
    # supported normalizations
    supported_normalizations = {
        'LayerNorm': nn.LayerNorm,
        'GroupNorm': nn.GroupNorm,
        'LayerNorm2d': LayerNorm2d,
        'BatchNorm1d': nn.BatchNorm1d,
        'BatchNorm2d': nn.BatchNorm2d,
        'BatchNorm3d': nn.BatchNorm3d,
        'SyncBatchNorm': nn.SyncBatchNorm,
        'InstanceNorm1d': nn.InstanceNorm1d,
        'InstanceNorm2d': nn.InstanceNorm2d,
        'InstanceNorm3d': nn.InstanceNorm3d,
    }
    norm_type = norm_cfg.pop('type')
    if norm_type== 'GroupNorm':
        normalization = supported_normalizations[norm_type](4,placeholder, **norm_cfg)
    else:
        normalization = supported_normalizations[norm_type](placeholder, **norm_cfg)
    return normalization

class SelfAttentionBlock_op(nn.Module):
    def __init__(self,  query_in_channels, transform_channels, out_channels, share_key_query,
                 query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm,
                 value_out_norm, matmul_norm, with_out_project, norm_cfg=None, act_cfg=None):
        super(SelfAttentionBlock_op, self).__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_query_num_convs=key_query_num_convs
        self.value_out_num_convs = value_out_num_convs
        # key wight

        # out project
        self.out_project = with_out_project

        # downsample
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.transform_channels = transform_channels

        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
            use_norm=key_query_norm,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.key_project=self.buildproject(in_channels=query_in_channels,out_channels=transform_channels,num_convs=self.key_query_num_convs,use_norm=True,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)

        self.value_project = self.buildproject(in_channels=query_in_channels, out_channels=transform_channels, num_convs=self.value_out_num_convs,
                                      use_norm=True, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.out_project = None
        if with_out_project:
            self.out_project = self.buildproject(
                    in_channels=transform_channels*2,
                    out_channels=query_in_channels,
                    num_convs=self.value_out_num_convs,
                    use_norm=True,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
    def buildproject(self, in_channels, out_channels, num_convs, use_norm, norm_cfg, act_cfg):
        if use_norm:
            convs = [nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )]
            for _ in range(num_convs - 1):
                convs.append(nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
                    BuildActivation(act_cfg),
                ))
        else:
            convs = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]
    '''forward'''
    def forward(self, query_feats):
        query = self.query_project(query_feats)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(query_feats)
        value = self.value_project(query_feats)

        value = value.permute(0, 2, 1).contiguous()
        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()


        if self.out_project is not None:
            context = torch.concat([context,query_feats],dim=1)
            context = self.out_project(context)
        return context
    '''build project'''

'''ImageLevelContext'''
class ImageLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False, align_corners=False, norm_cfg=None, act_cfg=None):
        super(ImageLevelContext, self).__init__()
        self.align_corners = align_corners
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
    '''forward'''
    def forward(self, x):
        x_global = self.global_avgpool(x)   ##全局平均池化
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il

class ImageLevelContext_3D(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False, align_corners=False, norm_cfg=None,act_cfg=None):
        super(ImageLevelContext_3D, self).__init__()
        self.align_corners = align_corners
        self.global_avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv3d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
    '''forward'''
    def forward(self, x):
        x_global = self.global_avgpool(x)   ##全局平均池化
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='trilinear', align_corners=self.align_corners)
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il

class PAM(nn.Module):
    def __init__(self, feats_channels, out_feats_channels, concat_input=False, norm_cfg=None, act_cfg=None):
        super(PAM, self).__init__()

        if concat_input:
            self.bottleneck = nn.Sequential(
                #nn.Conv3d(feats_channels*2, feats_channels*2, kernel_size=3, stride=1, padding=1, bias=False,groups=feats_channels),
                nn.Conv3d(feats_channels * 2, out_feats_channels, kernel_size=1, stride=1, padding=0),
                BuildNormalization(placeholder=out_feats_channels, norm_cfg=norm_cfg),
                BuildActivation(act_cfg),)

    def forward(self, x,preds):

        # print("dd",x.shape,preds.shape)
        inputs = x
        # print("intttt",inputs.type())

        batch_size, num_channels, h, w,d = x.size()
        num_classes = preds.size(1)   #psp ispp

        feats_sl = torch.zeros(batch_size, h*w*d, num_channels).type_as(x)
        feats_sl.float()
        for batch_idx in range(batch_size):
            # (C, H, W，d), (num_classes, H, W，d) --> (H*W, C), (H*W, num_classes)
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            # feats_iter.half()
            # print("ifff", feats_iter.type())
            feats_iter, preds_iter = feats_iter.reshape(num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1, 0).contiguous(), preds_iter.permute(1, 0).contiguous()

            # (H*W, )
            argmax = preds_iter.argmax(1)
            for clsid in range(num_classes):
                mask = (argmax == clsid)
                if mask.sum() == 0: continue
                feats_iter_cls = feats_iter[mask]

                preds_iter_cls = preds_iter[:, clsid][mask]

                weight = F.softmax(preds_iter_cls, dim=0)

                feats_iter_cls = feats_iter_cls*weight.unsqueeze(-1)
                feats_iter_cls = feats_iter_cls.mean(0)   #
                # print("feats1",feats_sl.type(),feats_iter_cls.type())
                if feats_sl.type() !=feats_iter_cls.type():
                    # print("feats2", feats_sl.type(), feats_iter_cls.type())
                    feats_iter_cls=feats_iter_cls.type_as(feats_sl)
                    # print("feats3", feats_sl.type(), feats_iter_cls.type())
                feats_sl[batch_idx][mask] = feats_iter_cls

        # feats_sl = self.self_attention(feats_sl)
        feats_sl = feats_sl.reshape(batch_size, h, w,d, num_channels)
        feats_sl = feats_sl.permute(0, 4, 1, 2,3).contiguous()
        feats_sl = inputs*feats_sl

        if hasattr(self, 'bottleneck'):
            # feats_sl = self.bottleneck(feats_sl) #cross attion
            feats_sl = self.bottleneck(torch.concat([feats_sl, inputs], dim=1))

        return feats_sl
if __name__ == "__main__":
    import torch as t
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    print('-----' * 5)
    rga = t.randn(2, 256, 8, 8, 8)
    rgb = t.randn(2,512,32,32)
    # rgb = t.randn(1, 3, 352, 480, 150)
    # net = Unet()
    # out = net(rga)
    # print(out.shape)
    # a = t.randn(2, 32, 128, 64, 64)
    slc_cfg = {'feats_channels': 256,
        'transform_channels': 256,
        'concat_input': True,
        'norm_cfg': {'type': 'InstanceNorm3d'},
        'act_cfg': {'type': 'ReLU', 'inplace': True},}
    # net = ImageLevelContext_3D(**ilc_cfg)
    net = PAM(**slc_cfg)
    flop = FlopCountAnalysis(net, rga)

    out = net(rga)
    # print(net)
    print("paramr", parameter_count_table(net))
    print(out.shape)