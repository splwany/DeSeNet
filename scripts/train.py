import argparse
import enum
import logging
import math
import os
import random
import scripts
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import imgviz
import cv2
from PIL import Image
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
root_path = str(Path(__file__).parent.parent.absolute())
sys.path.append(root_path)

from core.utils.mixed_datasets import create_mixed_dataloader
from core.utils.general import check_file, check_git_status, check_requirements, colorstr, increment_path, init_seeds, set_logging, xyxy2xywh
from core.utils.torch_utils import ModelEMA, select_device
from core.utils.plots import plot_one_box, colors
# from core.utils.wandb_logging.wandb_utils import check_wandb_resume

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, rank = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.global_rank,
    )

    # 目录
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # 保存运行的设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    
    # 配置
    plots = True
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)  # data 数据    
    
    de, se = data_dict['de'], data_dict['se']
    de_nc = int(de['nc'])  # 目标检测数据集类别数
    de_names = de['names']  # 目标检测数据集类别名
    assert len(de_names) == de_nc, '%g names found for nc=%g dataset in %s: de' % (len(de_names), de_nc, opt.data)  # check
    se_nc = int(se['nc'])  # 语义分割数据集类别数
    se_names = se['names']  # 语义分割数据集类别名
    assert len(se_names) == se_nc, '%g names found for nc=%g dataset in %s: se' % (len(se_names), se_nc, opt.data)  # check
    
    # 模型
    train_path = data_dict['train']
    test_path = data_dict['val']

    # 冻结参数
    # freeze = []  # 这里定义要冻结的参数名
    # for k, v in model.named_parameters():
    #     if k in freeze:
    #         print(f'freezing {k}')
    #         v.requires_grad = False
    #     else:
    #         v.requires_grad = True

    # 优化
    # nbs = 64  # nominal batch size (名义 batch_size)    
    # accumulate = max(round(nbs / total_batch_size), 1)  # 在优化前累积损失
    # hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # 按比例缩放 weight_decay
    # logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # for k, v in model.named_modules():
    #     if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
    #         pg2.append(v.bias)  # biases
    #     if isinstance(v, nn.BatchNorm2d):
    #         pg0.append(v.weight)  # no decay
    #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
    #         pg1.append(v.weight)  # apply decay

    # if opt.adam:
    #     optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # else:
    #     optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    # del pg0, pg1, pg2

    # EMA
    # ema = ModelEMA(model) if rank in [-1, 0] else None

    # 图像尺寸
    # gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    gs = 32
    imgsz, imgsz_test = 640, 640

    # DP 模式
    # if cuda and rank == -1 and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # if opt.sync_bn and cuda and rank != -1:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    #     logger.info('Using SyncBatchNorm()')

    for epoch in range(2):
        # Trainloader
        dataloader, dataset = create_mixed_dataloader(test_path, imgsz, batch_size, gs, opt,
                                                hyp=hyp, augment=True, rect=opt.rect, rank=rank,
                                                world_size=opt.world_size, workers=opt.workers,
                                                image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '), seed=3+rank+epoch)
        mlc = np.concatenate(dataset.det_labels, 0)[:, 0].max()  # max label class 标签中共有多少类
        nb = len(dataloader)  # batch 总数
        assert mlc < de_nc, f'标签类别数 {mlc} 超过 {opt.data} 中的 nc={de_nc}. nc 值的范围是 0-{de_nc - 1}'

        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        for i, (imgs, det_labels, seg_labels, paths, _) in pbar:  # imgs.shape == torch.Size([16, 3, 640, 640])
            out_imgs = imgs.permute(0, 2, 3, 1).numpy()
            out_segs = seg_labels.numpy()
            for j, (out_img, out_det, out_seg, path) in enumerate(zip(out_imgs, det_labels, out_segs, paths)):
                for det in out_det:
                    c = int(det[1])
                    label = de_names[c]
                    plot_one_box(det[2:], out_img, label=label, color=colors(c, True), line_thickness=3)
                # path_preffix = Path(path).stem
                path_preffix = f'{epoch}_{i}_{j}_'
                cv2.imwrite(f'runs/tmp/{path_preffix}.jpg', cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
                file_name = f'runs/tmp/{path_preffix}_label.png'
                if (out_seg > 0).any() and out_seg.min() >= 0 and out_seg.max() < 255:
                    lbl_pil = Image.fromarray(out_seg.astype(np.uint8), mode="P")
                    colormap = imgviz.label_colormap()
                    lbl_pil.putpalette(colormap.flatten())
                    lbl_pil.save(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='core/data/blind.yaml', help='data.yaml 路径')
    parser.add_argument('--hyp', type=str, default='core/hyp/scratch.yaml', help='超参数路径')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='所有GPU的总batch size')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--sync-bn', action='store_true', help='使用 SyncBatchNorm, 仅 DDP 模式可用')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP 参数, 不要改')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda 设备, 例如 0 或者 0,1,2,3 或者 cpu')
    parser.add_argument('--single-cls', action='store_true', help='将多类当作一类来训练')
    parser.add_argument('--workers', type=int, default=8, help='dataloader workers 的最大值')
    parser.add_argument('--project', default='runs/train', help='保存到 project/name')
    parser.add_argument('--name', default='exp', help='保存到 project/name')
    parser.add_argument('--exist-ok', action='store_true', help='project/name 可以存在, 不会新建')
    parser.add_argument('--quad', action='store_true', help='四分之一 dataloader')

    opt = parser.parse_args()

    # 设置 DDP 变量
    opt.world_size = int(os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ else 1)
    opt.global_rank = int(os.environ['RANK'] if 'RANK' in os.environ else -1)
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        # check_git_status()
        check_requirements(exclude=('pycocotools', 'thop'))
    
    # 恢复，回到之前位置
    opt.data, opt.hyp = check_file(opt.data), check_file(opt.hyp)  # 检查文件
    # assert len(opt.cfg) or len(opt.weights), '--cfg 和 --weights 必须指定'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP 模式
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        # distributed backend
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size 必须是GPU个数的倍数'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        opt.batch_size = opt.total_batch_size // opt.world_size
    
    # 超参数
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # 加载超参数

    # 训练
    logger.info(opt)
    tb_writer = None  # 初始化 loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(
            f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # tensorboard
    train(hyp, opt, device, tb_writer)
