# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import logging
import math
import warnings
from copy import copy
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.general import is_ascii
from PIL import Image
from torch.cuda import amp
from core.utils.general import (colorstr, increment_path, make_divisible,
                           non_max_suppression, save_one_box, scale_coords,
                           xyxy2xywh)
from core.utils.mixed_datasets import exif_transpose, letterbox
from core.utils.plots import Annotator, colors, plot_one_box
from core.utils.torch_utils import time_sync

LOGGER = logging.getLogger(__name__)


def autopad(k: Union[int, List[int]], p=None) -> Union[int, List[int]]:  # kernel, padding
    # Pad to 'same'
    if p is None:
        if isinstance(k, int):
            p = k // 2
        elif isinstance(k, list):
            p = [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        ap = autopad(k, p)
        assert isinstance(ap, int)
        self.conv = nn.Conv2d(c1, c2, k, s, ap, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) if x[0][0].numel() > 1 else self.conv(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class ARM(nn.Module):   # AttentionRefinementModule
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(ARM, self).__init__()
        self.conv = Conv(in_chan, out_chan, k=3, s=1, p=None)  #　Conv 自动padding
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),  # ARM的SE带bn不带act
                                               Conv(out_chan, out_chan, k=1, s=1,act=False),   # 注意ARM的SE处用了BN，FFM没用，SE用了BN的模型training时不支持单个样本，对应改了两处，一是yolo.py构造好跑一次改成了(2,3,256,256)
                                               nn.Sigmoid()                 # 二是train.py的batch开头加了一句单样本时候continue(分割loader容易加droplast，但是检测loader出现地方太多没分mode不好改)
                                               )            

    def forward(self, x):
        feat = self.conv(x)  # 先3*3卷积一次
        atten = self.channel_attention(feat)  # SE
        return torch.mul(feat, atten)


# 上海交大 FFM
class FFM(nn.Module):  # FeatureFusionModule  reduction用来控制瓶颈结构
    def __init__(self, in_chan, out_chan, reduction=1, is_cat=True, k=1):
        super(FFM, self).__init__()
        self.convblk = Conv(in_chan, out_chan, k=k, s=1)  ## 注意力处用了１＊１瓶颈，两个卷积都不带bn,一个带普通激活，一个sigmoid
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(out_chan, out_chan//reduction,
                                                         kernel_size = 1, stride = 1, padding = 0, bias = False),
                                               nn.SiLU(inplace=True),
                                               nn.Conv2d(out_chan//reduction, out_chan,
                                                         kernel_size = 1, stride = 1, padding = 0, bias = False),
                                               nn.Sigmoid(),
                                            )
        self.is_cat = is_cat

    def forward(self, fspfcp):  #空间, 语义两个张量用[]包裹送入模块，为了方便Sequential
        fcat = torch.cat(fspfcp, dim=1) if self.is_cat else fspfcp
        feat = self.convblk(fcat)
        atten = self.channel_attention(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


# 宋雪娜 FFM
# class FFM(nn.Module):
#     # FeatureFusionModule reduction用来控制瓶颈结构
#     def __init__(self, c1, c2, reduction=1, is_cat=True, k=1):
#         super(FFM, self).__init__()
#         self.conv1x1 = ConvBNReLU(c1, c2, k=k, s=1, p=None)
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             ConvBNReLU(c2, c2//reduction, 1, 1, None),
#             ConvBNReLU(c2//reduction, c2, 1, 1, None),
#             nn.Sigmoid(),
#         )
#         self.is_cat = is_cat

#     def forward(self, x):
#         # 删除FFM的forward里的cat语句，直接使用yolo原结构中concat后的结果作为输入
#         fusion = torch.cat(x, dim=1) if self.is_cat else x
#         out = self.conv1x1(x)
#         attention = self.channel_attention(out)
#         out = out + out * attention
#         return out


class ASPP(nn.Module):  # ASPP，原版没有hid，为了灵活性方便砍通道增加hid，hid和out一样就是原版
    def __init__(self, in_planes, out_planes, d=[3, 6, 9], has_global=True, map_reduce=4):
        super(ASPP, self).__init__()
        self.has_global = has_global
        self.hid = in_planes//map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, self.hid, k=1, s=1),
                )
        self.branch1 = nn.Sequential(
                nn.Conv2d(in_planes, self.hid, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(in_planes, self.hid, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                nn.Conv2d(in_planes, self.hid, kernel_size=3, stride=1, padding=d[2], dilation=d[2], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        if self.has_global:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(in_planes, self.hid, k=1),
                )
        self.ConvLinear = Conv(int(5*self.hid) if has_global else int(4*self.hid), out_planes, k=1, s=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.has_global:
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3],1))
            return out
        else:
            x4 = F.interpolate(self.branch4(x), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3,x4],1))
            return out


class ASPPs(nn.Module):  # 空洞卷积前先用1*1砍通道到目标（即相比上面版本空洞卷积的输入通道减少，一个1*1统一砍通道试过效果不好，每个分支1*1独立,1*1分支改3*3）
    def __init__(self, in_planes, out_planes, d=[3, 6, 9], has_global=True, map_reduce=4):
        super(ASPPs, self).__init__()
        self.has_global = has_global
        self.hid = in_planes//map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),
                Conv(self.hid, self.hid, k=3, s=1),
                )
        self.branch1 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),
                nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),    
                nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                Conv(in_planes, self.hid, k=1),
                nn.Conv2d(self.hid, self.hid, kernel_size=3, stride=1, padding=d[2], dilation=d[2], bias=False),
                nn.BatchNorm2d(self.hid),
                nn.SiLU()    
                )
        if self.has_global:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(in_planes, self.hid, k=1),
                )
        self.ConvLinear = Conv(int(5*self.hid) if has_global else int(4*self.hid), out_planes, k=1, s=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.has_global:
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3],1))
            return out
        else:
            x4 = F.interpolate(self.branch4(x), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3,x4],1))
            return out


class DAPPM(nn.Module):
    """
    https://github.com/ydhongHIT/DDRNet，只换了激活函数，原仓库代码每个Block里Conv,BN,Activation的顺序写法很非主流,这种非主流写法应该也是考虑了两个层相加后再进行BN和激活
    使用注意，若遵照原作者用法，1、此模块前一个Block只Conv，不BN和激活（因为每个scale pooling后BN和激活）；
                           2、此模块后一个Block先BN和激活再接其他卷积层（模块结束后与高分辨率相加后统一BN和激活，与之相加的高分辨率的上一Block最后也不带BN和激活）
    """
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes * 5),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    nn.BatchNorm2d(inplanes),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 


# 和ASPPs类似(初衷都是为了砍ASPP计算量，这个模块砍中间和输入通道增加3*3卷积补偿;ASPPs砍中间和输入通道，没有多的操作，同延时下可以少砍一点)
class RFB1(nn.Module):  # 魔改ASPP和RFB,这个模块其实长得更像ASPP,相比RFB少shortcut,３＊３没有宽高分离,d没有按照RFB设置;相比ASPP多了1*1砍输入通道和3*3卷积
    def __init__(self, in_planes, out_planes, map_reduce=4, d=[3, 5, 7], has_global=False):
        super(RFB1, self).__init__()
        self.out_channels = out_planes
        self.has_global = has_global
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1)
                )
        self.branch1 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1),
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1),
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=5, s=1),
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[2], dilation=d[2], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()    
                )
        if self.has_global:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(in_planes, inter_planes, k=1),
                )
        self.Fusion = Conv(int(5*inter_planes) if has_global else int(4*inter_planes), out_planes, k=1, s=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if not self.has_global:
            out = self.Fusion(torch.cat([x0,x1,x2,x3], 1))
            return out
        else:
            x4 = F.interpolate(self.branch4(x), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.Fusion(torch.cat([x0,x1,x2,x3,x4],1))
            return out


class RFB2(nn.Module):  # 魔改模块,除了历史遗留(改完训练模型精度不错，不想改名重训)名字叫RFB，其实和RFB没啥关系了(参考deeplabv3的反面级联结构，也有点像CSP，由于是级联，d设置参考论文HDC避免网格效应)实验效果不错，能满足较好非线性、扩大感受野、多尺度融合的初衷(在bise中单个精度和多个其他模块组合差不多，速度和C3相近比ASPP之类的快)
    def __init__(self, in_planes, out_planes, map_reduce=4, d=[2, 3], has_global=False):  # 第一个3*3的d相当于1，典型的设置1,2,3; 1,2,5; 1,3,5
        super(RFB2, self).__init__()
        self.out_channels = out_planes
        self.has_global = has_global
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),
                Conv(inter_planes, inter_planes, k=3, s=1)
                )
        self.branch1 = nn.Sequential(
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()    
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
                nn.BatchNorm2d(inter_planes),
                nn.SiLU()                    
                )
        self.branch3 = nn.Sequential(
                Conv(in_planes, inter_planes, k=1, s=1),  
                )
        if self.has_global:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(inter_planes, inter_planes, k=1),
                )
        self.ConvLinear = Conv(int((5 if has_global else 4) * inter_planes), out_planes, k=1, s=1)

    def forward(self, x):  # 思路就是rate逐渐递进的空洞卷积连续卷扩大感受野避免使用rate太大的卷积(级联注意rate要满足HDC公式且不应该有非1公倍数，空洞卷积网格效应)，多个并联获取多尺度特征
        x3 = self.branch3(x)  # １＊１是独立的　类似C3，区别在于全部都会cat
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        if not self.has_global:
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3], 1))
        else:
            x4 = F.interpolate(self.branch4(x2), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3,x4], 1))
        return out


class ACSP(nn.Module):  # 在RFB2的基础上加入PyramidPooling，并更改branch3的连接
    def __init__(self, in_planes, out_planes, map_reduce=4, d=[2, 3], has_global=False):  # 第一个3*3的d相当于1，典型的设置1,2,3; 1,2,5; 1,3,5
        super().__init__()
        self.out_channels = out_planes
        self.has_global = has_global
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1, s=1),
            Conv(inter_planes, inter_planes, k=3, s=1)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
            nn.BatchNorm2d(inter_planes),
            nn.SiLU()    
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
            nn.BatchNorm2d(inter_planes),
            nn.SiLU()                    
        )
        if self.has_global:
            self.branch_global = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(inter_planes, inter_planes, k=1),
            )
        self.ConvLinear = Conv(int((4 if has_global else 3) * inter_planes), out_planes, k=1, s=1)

    def forward(self, x):  # 思路就是rate逐渐递进的空洞卷积连续卷扩大感受野避免使用rate太大的卷积(级联注意rate要满足HDC公式且不应该有非1公倍数，空洞卷积网格效应)，多个并联获取多尺度特征
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        if not self.has_global:
            out = self.ConvLinear(torch.cat([x0,x1,x2], 1))
        else:
            x3 = F.interpolate(self.branch_global(x2), (x.shape[2], x.shape[3]), mode='nearest')  # 全局
            out = self.ConvLinear(torch.cat([x0,x1,x2,x3], 1))
        return out


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, k=[1, 2, 3, 6], short_cut=False):
        super(PyramidPooling, self).__init__()
        self.short_cut = short_cut
        
        self.pool1 = nn.AdaptiveAvgPool2d(k[0])
        self.pool2 = nn.AdaptiveAvgPool2d(k[1])
        self.pool3 = nn.AdaptiveAvgPool2d(k[2])
        self.pool4 = nn.AdaptiveAvgPool2d(k[3])

        out_channels = in_channels//4
        self.conv1 = Conv(in_channels, out_channels, k=1)
        self.conv2 = Conv(in_channels, out_channels, k=1)
        self.conv3 = Conv(in_channels, out_channels, k=1)
        self.conv4 = Conv(in_channels, out_channels, k=1)

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=True)

        return torch.cat((x, feat1, feat2, feat3, feat4), 1) if self.short_cut else torch.cat((feat1, feat2, feat3, feat4), 1)
    

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
    
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False) if s == 2 else nn.Identity())
    
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# class NMS(nn.Module):
#     # Non-Maximum Suppression (NMS) module
#     conf = 0.25  # confidence threshold
#     iou = 0.45  # IoU threshold
#     classes = None  # (optional list) filter by class
#     max_det = 1000  # maximum number of detections per image

#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return non_max_suppression(x[0], self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(str(im), stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs: List[np.ndarray], pred: List[torch.Tensor], files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        if times is not None:
            self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, pil=not self.ascii)
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            # plot_one_box(box, im, label=label, color=colors(cls))
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                str += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: {self.t[0]:.1f}ms pre-process, {self.t[1]:.1f}ms inference, {self.t[2]:.1f}ms NMS per image at shape {tuple(self.s)}')

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], names=self.names, shape=self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k: int=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        ap = autopad(k, p)
        assert isinstance(ap, int)
        self.conv = nn.Conv2d(c1, c2, k, s, ap, groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


# 宋雪娜新添加 FFM 需要用到的 ------------------------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    # sxn：与Conv区别：激活函数不同
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, relu6=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
