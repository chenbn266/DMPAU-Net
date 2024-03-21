import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from typing import List, Optional, Sequence, Tuple, Union
from monai.networks.nets import DynUNet
from moudels.backbones.swintransformer import SwinTransformerSys3D
from moudels.backbones.unet_swin import Swin_Unet
from moudels.backbones.you import Unet
class cascade_vtunet(nn.Module):
    def __init__(self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.trans_bias = trans_bias

        self.filters = filters

        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num

        self.unet = Unet(c=4, n=16, dropout=0.5, norm='gn', num_classes=3)
        self.swin_unet = Swin_Unet(c=7, n=16, dropout=0.5, norm='gn', num_classes=3)
    def forward(self,x):
        x1 = self.unet(x)
        # print("x1,x",x1.shape, x.shape)
        x2 = torch.cat([x,x1],1)

        x3 = self.swin_unet(x2)
        # print("x.shape x3.shape",x.shape,x3.shape)
        return x3
def get_unet_params():

    patch_size, spacings = [64, 64, 64], [1.0, 1.0, 1.0]
    strides, kernels, sizes = [], [], patch_size[:]
    while True:
        spacing_ratio = [spacing / min(spacings) for spacing in spacings]
        stride = [
            2 if ratio <= 2 and size >= 2 * 2 else 1 for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
        if len(strides) == 6:
            break
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides, patch_size

if __name__ == "__main__":
    import torch as t

    print('-----' * 5)
    rga = t.randn(1, 4, 128, 128, 128)
    # rga = t.randn(1, 32, 64, 64, 64)
    # rga = t.randn(1,64,32,32,32)
    kernels,strides,patch_size=get_unet_params()
    net = cascade_vtunet(3,4,3,
                  kernels,strides,strides[1:],
                  filters=[64, 96, 128, 192, 256, 384 ,512],
                  norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=True,
            deep_supr_num=2,
            res_block=False,
            trans_bias=True,

                  )
    # swin =cascade_vtunet(c=4, n=16, dropout=0.5, norm='gn', num_classes=3)

    print(net(rga).shape)