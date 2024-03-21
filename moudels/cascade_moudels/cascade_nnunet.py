import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from typing import List, Optional, Sequence, Tuple, Union
from monai.networks.nets import DynUNet

class cascade_nnunet(nn.Module):
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

        self.model1 = DynUNet(
            self.spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            strides,
            strides[1:],
            filters=self.filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=self.deep_supervision,
            deep_supr_num=self.deep_supr_num,
            res_block=res_block,
            trans_bias=True,
        )
        self.model2 = DynUNet(
            self.spatial_dims,
            in_channels*2-1,
            out_channels,
            kernel_size,
            strides,
            strides[1:],
            filters=self.filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=self.deep_supervision,
            deep_supr_num=self.deep_supr_num,
            res_block=res_block,
            trans_bias=True,
        )

    def forward(self,x):
        x1 = self.model1(x)
        print("x1,x",x1.shape, x.shape)

        y = x1[:,0,:,:,:]
        print("y",y.shape)
        x2 = torch.cat([x,y],1)

        x3 = self.model2(x2)
        print("x2.shape x3.shape",x2.shape,x3.shape)
        return x3

