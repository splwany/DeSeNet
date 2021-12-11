"""YOLOv5-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # to run '$ python *.py' files in subdirectories
ROOT = ROOT.relative_to(Path.cwd())  # relative

from core.models.common import *
from core.models.experimental import *
from core.utils.autoanchor import check_anchor_order
from core.utils.general import make_divisible, check_file, set_logging
from core.utils.plots import feature_visualization
from core.utils.torch_utils import time_sync, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)


class SegMaskBiSe(nn.Module):  # 配置文件输入[16, 19, 22]通道无效
    def __init__(self, n_segcls=19, n=1, c_hid=256, shortcut=False, ch=()):  # n是C3的, c_hid是C3的输出通道数(接口保留了,没有使用,可用子模块控制s,m,l加深加宽)
        super().__init__()
        self.c_in8 = ch[0]  # 16 Lab实验用4更好，但是BiSe实验用16更好（原因可能在1/8通道一个48一个128）
        self.c_in16 = ch[1]  # 19
        self.c_in32 = ch[2]  # 22
        self.c_out = n_segcls

        self.m8 = nn.Sequential(  # 未采用双流结构
            Conv(self.c_in8, 128, k=1, s=1),
        )
        self.m16 = nn.Sequential(
            RFB2(self.c_in16, 128, map_reduce=4, d=[2,3], has_global=False),  # 魔改模块(和RFB没啥关系了,原则是增强分割入口非线性,同时扩大感受野和兼顾多尺度)，实验速度精度效果还不错
            # Attention(128),  # 可选，这层与1/32up相加，有相加处用Attention也是BiSeNet的ARM模块设计的初衷。前面有复杂模块，Attention就够了，没必要用ARM多个3*3计算量，核心目的是一样的
            # ARM(128, 128),
        )
        self.m32 = nn.Sequential(
            RFB2(self.c_in32, 128, map_reduce=8, d=[2,3], has_global=True),  # 舍弃原GP，在1/32(和1/16，可选)处加全局特征
            # Attention(128),  # 改变了global特征的获取方式，这层不用和globel特征相加，因此没必要用ARM或者Attention
            # ARM(128, 128),
        )
        # self.GP = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     Conv(self.c_in32, 128, k=1),
        # )
        self.up16 = nn.Sequential(
            Conv(128, 128, 3),  # refine论文源码每次up后一个3*3refine，降低计算量放在前
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.up32 = nn.Sequential(
            Conv(128, 128, 3),  # refine
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.out = nn.Sequential(
            FFM(256, 256, k=3),  
            nn.Dropout(0.1),  # 最后一层改用3*3，我认为用dropout不合适（dropout对3*3响应空间维度形成遮挡），改为dropout2d（随机整个通道置０增强特征图独立性，空间上不遮挡）
            nn.Conv2d(256, self.c_out, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        # 辅助分割头，训练用，推理丢弃
        self.aux16 = nn.Sequential(
            Conv(128, 128, 3),
            nn.Conv2d(128, self.c_out, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.aux32 = nn.Sequential(
            Conv(128, 128, 3),
            nn.Conv2d(128, self.c_out, kernel_size=1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        # GP = self.GP(x[2])  # 改成直接用广播机制加 F.interpolate(self.GP(x[2]), (x[2].shape[2], x[2].shape[3]), mode='nearest')  # 全局
        feat3 = self.up32(self.m32(x[2]))  #  + GP) 
        feat2 = self.up16(self.m16(x[1]) + feat3)
        feat1 = [self.m8(x[0]), feat2]
        return self.out(feat1) if not self.training else [self.out(feat1), self.aux16(feat2), self.aux32(feat3)]


# DeepLabV3+的encoder-decoder结构其实只涨了1个点(VOC上)，启示是ASPP放在1/16图上结合浅层图也能有很好的效果(如果放在1/8图是不会考虑在此模型尝试ASPP的,太重了,放在1/32试验过精度掉了,延时也没下去很多)
# 模仿DeepLabV3+(论文1/4和1/16)　但是YOLO的1/4图太过于浅且通道太少(s只有64,deeplab的backbone常有256以上所以1*1降维)而且1/4最后用3*3 refine计算量太大,这里取1/8和1/16
# 融合部分加了FFM(k改3)，deeplabv3+是两层3*3保持256通道（太奢侈），深浅并联融合第一层最好是3*3
# deeplabv3+论文经验是编码器解码器结构中，解码部分使用更少的浅层通道利于学习(论文48，32或64也接近，论文提了VOC当中用全局后提升，citys用全局后下降，这里没有用全局)
class SegMaskLab(nn.Module):  #   配置文件[3, 16, 19, 22], 通道配置无效
    def __init__(self, n_segcls=19, n=1, c_hid=256, shortcut=False, ch=()):  # n此处用于控制ASPP的map_reduce,配置文件写3, c_hid是输出通道数配置文件写256
        super().__init__()
        self.c_detail = ch[0]  # 4 YOLO的FPN是cat不是add，16cat了完整的4，理论上可以学出来，然而直接用４效果略好于16(同cat后1*1包含了add却并不总是比add好，问题在正则而不是容量)。
        self.c_in16 = ch[1]  # 19
        self.c_out = n_segcls
        # 实验效果细节层４>16, 使用1/8，没像deeplabv3+原文一样直接用1/4（l等大模型追求精度可以考虑用1/4相应的我认为融合层也该增加为两个3*3同原文）
        self.detail = nn.Sequential(Conv(self.c_detail, 48, k=1),
            Conv(48, 48, k=3),
        )
        self.encoder = nn.Sequential(
            # hid砍得越少精度越高(这里问题在容量)，maep_reduce=1相当于标准ASPP
            # 未使用全局，一方面遵照论文，一方面用了全局后出现边界破碎的情况
            Conv(self.c_in16, c_hid*2, k=1),
            ASPP(c_hid*2, 256, d=[3, 6, 9], has_global=False, map_reduce=5-n),  # ASPP确实好，但是太重了，砍到了1/4通道 s:5-1=4, m:5-2=3, l:5-3=2
            # 这两个都是ASPP的替代品, ASPP也有一个问题，光一个ASPP不够深，ASPPs和RFB1中间输入一起砍，ASPPs砍完可以选择前面加其他模块，RFB1砍后增加了3*3和5*5
            # ASPPs(256, 256, d=[4, 7, 10], has_global=False, map_reduce=5-n), # 
            # RFB1(self.c_in16, 256, d=[3, 5, 7], has_global=False, map_reduce=max(4-n, 2)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder = nn.Sequential(
            # 原论文两个3*3保持256(文中实验表示保持256最重要，其次是3*3)，此处为了速度还是得砍到128(第一个融合处想继续用3*3保证深浅融合效果)
            FFM(256+48, 256, k=1, is_cat=True),  # 融合用bisenet的配置
            Conv(256, c_hid, k=3),  # 经验是不管多宽，k取3还是1，用三层融合输出(有浅层融合)
            nn.Conv2d(c_hid, self.c_out, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        feat16 = self.encoder(x[1])  # 1/16主语义
        feat8 = self.detail(x[0])  # 1/8浅层
        return self.decoder([feat8, feat16])


# 一个性能不错的分割头140+FPS，验证集72.7~73.0,把1.5改成1.0则是72.4到72.7
# SPP增大了感受野，也提高了多尺度但还不够(我认为比起ASPP等的差距是本backbone和指标体现不出的,在数据集外的图上可视化能体现)，1/8比较大，SPP比较小，没有更大感受野
class SegMaskBase(nn.Module):
    def __init__(self, n_segcls=19, n=1, c_hid=256, shortcut=False, ch=()):  # n是C3的, c_hid是C3的输出通道数
        super().__init__()
        self.c_in = ch[0]  # 此版本Head暂时只有一层输入
        self.c_out = n_segcls
        self.m = nn.Sequential(
            C3(c1=self.c_in, c2=c_hid, n=n, shortcut=shortcut, g=1, e=0.5),
            # SPP(c_hid, c_hid, k=(5, 9, 13)),
            C3SPP(c1=c_hid, c2=int(c_hid*1.5), k=(5, 9, 13), g=1, e=0.5),

            #C3(c1=c_hid, c2=c_hid, n=n, shortcut=shortcut, g=1, e=0.5),
            #Conv(c1=c_hid, c2=c_hid, k=1, s=1),
            nn.Dropout(0.1, True),
            nn.Conv2d(int(c_hid*1.5), self.c_out, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), groups=1, bias=False),  # 后续几个头实验表明最后一层kernel还是1*1略好, base没有重训
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.m(x[0])  # self.up(self.conv(self.c3(x[0])))


class SegMaskPSP(nn.Module):  # PSP头，多了RFB2和FFM，同样砍了通道数，没找到合适的位置加辅助损失，因此放弃辅助损失
    def __init__(self, n_segcls=19, n=1, c_hid=256, shortcut=False, ch=()):  # n是C3的, (接口保留了,没有使用)c_hid是隐藏层输出通道数（注意配置文件s*0.5,m*0.75,l*1）
        super().__init__()
        self.c_in8 = ch[0]  # 16  # 用16,19,22宁可在融合处加深耗费一些时间，检测会涨点分割也很好。严格的消融实验证明用17,20,23分割可能还会微涨，但检测会掉３个点以上，所有头如此
        self.c_in16 = ch[1]  # 19
        self.c_in32 = ch[2]  # 22
        # self.c_aux = ch[0]  # 辅助损失  找不到合适地方放辅助，放弃
        self.c_out = n_segcls
        
        self.m8 = nn.Sequential(
            Conv(self.c_in8, c_hid, k=1),
        )
        self.m16 = nn.Sequential(
            Conv(self.c_in16, c_hid, k=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.m32 = nn.Sequential(
            Conv(self.c_in32, c_hid, k=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        
        # 注意配置文件通道写256,此时s模型c_hid＝128
        self.out = nn.Sequential(  # 实验表明引入较浅非线性不太强的层做分割会退化成检测的辅助(分割会相对低如72退到70,71，检测会明显升高)，PP前应加入非线性强一点的层并适当扩大感受野
            RFB2(c_hid*3, c_hid, d=[2,3], map_reduce=6),  # 3*128//6=64　RFB2和RFB无关，仅仅是历史遗留命名(训完与训练模型效果不错就没有改名重训了)
            PyramidPooling(c_hid, k=[1, 2, 3, 6], short_cut=True),  # 按原文1,2,3,6，PSP加全局更好，但是ASPP加了全局后出现边界破碎
            FFM(c_hid*2, c_hid, k=3, is_cat=False),  # FFM改用k=3, 相应的砍掉部分通道降低计算量(原则就是差距大的融合哪怕砍通道第一层也最好用3*3卷积，FFM融合效果又比一般卷积好，除base头外其他头都遵循这种融合方式)
            nn.Conv2d(c_hid, self.c_out, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        
        # self.aux = nn.Sequential(
        #     Conv(self.c_aux, 256, 3),  
        #     nn.Dropout(0.1, False), 
        #     nn.Conv2d(256, self.c_out, kernel_size=1),
        #     nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        # )

    def forward(self, x):
        # 这个头三层融合输入做过消融实验，单独16:72.6 三层融合:73.5, 建议所有用1/8的头都采用三层融合，在Lab的实验显示三层融合的1/16输入也有增长
        feat = torch.cat([self.m8(x[0]), self.m16(x[1]), self.m32(x[2])], 1)
        # return self.out(feat) if not self.training else [self.out(feat), self.aux(x[0])]
        return self.out(feat)


class SegMaskPSP2(nn.Module):  # 自己优化的结构，移动连接顺序
    def __init__(self, n_segcls=19, n=1, c_hid=256, ch=()):
        super().__init__()
        assert len(ch) == 3
        self.c_in8, self.c_in16, self.c_in32 = ch  # 16、19、22层的输出通道数
        self.c_out = n_segcls
        
        self.m8 = nn.Sequential(
            Conv(self.c_in8, c_hid, k=1)
        )
        self.m16 = nn.Sequential(
            Conv(self.c_in16, c_hid, k=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.m32 = nn.Sequential(
            Conv(self.c_in32, c_hid, k=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        
        map_reduce = 6
        self.spatial_path = ACSP(c_hid*3, c_hid, d=[2, 3], map_reduce=map_reduce, has_global=True)
        self.context_path = nn.Sequential(
            Conv(c_hid*3, c_hid//map_reduce, k=1, s=1),
            PyramidPooling(c_hid//map_reduce, k=[1,2,3,6], short_cut=False)
        )
        self.ffm = FFM(c_hid+c_hid//map_reduce//4*4, c_hid, k=3, is_cat=True)
        
        self.out = nn.Sequential(
            nn.Conv2d(c_hid, self.c_out, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        feat = torch.cat([self.m8(x[0]), self.m16(x[1]), self.m32(x[2])], 1)
        feat = self.ffm([self.spatial_path(feat), self.context_path(feat)])
        return self.out(feat)


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
                
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, 'r', encoding='utf-8') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['de_nc']:
            LOGGER.info(f"Overriding model.yaml de_nc={self.yaml['de_nc']} with de_nc={nc}")
            self.yaml['de_nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.save.append(24)  # TODO 增加记录分割层
        self.de_names = [str(i) for i in range(self.yaml['de_nc'])]  # default names
        self.se_names = [str(i) for i in range(self.yaml['se_nc'])]
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(2, ch, s, s))[0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)  # 初始化, 看代码只初始化了BN和激活函数,跳过了卷积层
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            # 调好输入每层都是直接跑，detect是最后一层，for循环最后一个自然是detect结果
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output 解析时self.save记录了需要保存的那些层（后续层的输入会用到），仅保存这些层再输出即可
            if visualize:
                feature_visualization(x, m._type, m.i, save_dir=visualize)
        return x, y[-2]  # 目标检测，语义分割

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p
    
    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m._type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s} Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    # def nms(self, mode=True):  # add or remove NMS module
    #     present = type(self.model[-1]) is NMS  # last layer is NMS
    #     if mode and not present:
    #         LOGGER.info('Adding NMS... ')
    #         m = NMS()  # module
    #         m.f = -1  # from
    #         m.i = self.model[-1].i + 1  # index
    #         self.model.add_module(name='%s' % m.i, module=m)  # add
    #         self.eval()
    #     elif not mode and present:
    #         LOGGER.info('Removing NMS... ')
    #         self.model = self.model[:-1]  # remove
    #     return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'de_nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, de_nc, se_nc, gd, gw = d['anchors'], d['de_nc'], d['se_nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (de_nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1  # 置 1 表示深度对这三个模块是控制子结构重复，而不是本身重复
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in [SegMaskLab, SegMaskBase, SegMaskPSP, SegMaskPSP2]:  # 语义分割头
            args[1] = max(round(args[1] * gd), 1) if args[1] > 1 else args[1]  # SegMask 中 C3 的 n (Lab 里用来控制 ASPP 砍多少通道)
            args[2] = make_divisible(args[2] * gw, 8)  # SegMask C3 (或其他可放缩子结构) 的输出通道数
            args.append([ch[x] for x in f])
            # n = 1  # 不用设 1 了，SegMask 自己配置文件的 n 永远为 1
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        params_number = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_._type, m_.np = i, f, t, params_number  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, params_number, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=ROOT / 'yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
