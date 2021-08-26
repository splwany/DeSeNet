import datetime
import logging
import math
import platform
from contextlib import contextmanager
from pathlib import Path
import subprocess
from copy import deepcopy
import os
from typing import no_type_check_decorator

import torch
import torch.distributed as distributed
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # 计算 FLOPS
except ImportError:
    thop = None

logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """ 用来同步进程的装饰器
    在分布式训练时，让其他进程等待主进程执行完一些操作后再一起运行
    """
    if local_rank not in [-1, 0]:
        distributed.barrier()
    yield
    if local_rank == 0:
        distributed.barrier()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'DeSeNet 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def is_parallel(model):
    # 如果模型是 DP 或 DDP 类型，返回 True
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """将一个模型解除并行：如果模型是 DP 或 DDP 类型，返回一个单 GPU 模型"""
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    """把 b 中不以 '_' 开头的属性和方法复制到 a。include 为白名单，exclude为黑名单"""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ [模型指数移动平均](https://github.com/rwightman/pytorch-image-models)保持模型 state_dict（参数和缓冲区） 中所有内容的移动平均值。
    这是为了允许[这里](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)介绍的功能，
    一些训练计划要想取得好成绩，一个平滑版本的权重是十分必要的。
    这个类安置在模型初始化、GPU分配和分布式训练封装等一系列操作中是敏感的。
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        model: 模型
        decay: 衰减
        update: EMA 更新的个数，默认为 0
        """

        # 创建 EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 指数下降衰减 (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """更新 EMA 的参数们"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # 模型的 state_dict
            for k, v in self.ema.state_dict().items():
                assert isinstance(v, torch.Tensor)
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """更新 EMA attributes"""
        copy_attr(self.ema, model, include, exclude)
