import argparse
from collections import OrderedDict
from core.utils.google_utils import attempt_download
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import imgviz
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
if str(FILE.parents[1] not in sys.path):
    sys.path.append(str(FILE.parents[1]))  # add yolov5/ to path

from core.models.yolo import Model
from core.utils.autoanchor import check_anchors
from core.utils.mixed_datasets import create_mixed_dataloader
from core.utils.general import (check_dataset, check_file, check_git_status, check_img_size, check_suffix,
                                check_requirements, colorstr, get_latest_run, one_cycle,
                                increment_path, init_seeds, set_logging,
                                xywhn2xyxy, xyxy2xywh, methods)
from core.utils.plots import colors, plot_labels, plot_one_box
from core.utils.torch_utils import ModelEMA, select_device, de_parallel, intersect_dicts, torch_distributed_zero_first
from core.utils.wandb_logging.wandb_utils import check_wandb_resume
from core.utils.metrics import fitness
from core.utils.loggers import Loggers
from core.utils.callbacks import Callbacks

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp, opt, device, callbacks):
    # 从 opt 中获取相关参数
    save_dir, epochs, batch_size, weights, single_cls = Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls
    data, cfg, resume, workers, freeze = opt.data, opt.cfg, opt.resume, opt.workers, opt.freeze
    
    # 超参数
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # 加载 hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # 目录
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # 保存运行的设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers()  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
        # 注册 actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # 配置
    plots = True
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val'] 

    de = data_dict['de']
    de_nc = 1 if single_cls else int(de['nc'])  # 目标检测数据集类别数
    de_names = ['item'] if single_cls and len(de['names'] != 1) else de['names']  # 目标检测数据集类别名
    assert len(de_names) == de_nc, f'{len(de_names)} names found for nc={de_nc} detection dataset in {data}'  # check
    
    se = data_dict['se']
    se_nc = 1 if single_cls else int(se['nc'])  # 语义分割数据集类别数
    se_names = ['item'] if single_cls and len(se['names'] != 1) else se['names']  # 语义分割数据集类别名
    assert len(se_names) == se_nc, f'{len(se_names)} names found for nc={se_nc} segmentation dataset in {data}'  # check
    
    # 模型
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    ckpt = dict()
    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # 如果本地没找到则下载
        ckpt = torch.load(weights, map_location=device)  # 加载检查点
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=de_nc, anchors=hyp.get('anchors')).to(device)  # 创建模型
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # 排除的属性
        csd = ckpt['model'].float().state_dict()  # 检查点的 state_dict (FP32)
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # 取 csd 和 state_dict 的交集，且不在 exclude 中
        model.load_state_dict(csd, strict=False)  # 加载
        LOGGER.info(f'从 {weights} 转移了 {len(csd)}/{len(model.state_dict())} 项')  # 报告
    else:
        model = Model(cfg, ch=3, nc=de_nc, anchors=hyp.get('anchors')).to(device)

    # 冻结参数
    freeze = [f'model.{x}.' for x in range(freeze)]  # 这里定义要冻结的层
    for k, v in model.named_parameters():
        if any(x in k for x in freeze):
            print(f'正在冻结 {k}')
            v.requires_grad = False
        else:
            v.requires_grad = True

    # 优化
    nbs = 64  # nominal batch size (名义 batch_size)    
    accumulate = max(round(nbs / batch_size), 1)  # 在优化前累积损失
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 按比例缩放 weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('Optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # 学习率调节器
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1 -> hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # 恢复
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 优化器
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            saved_ema: OrderedDict[str, torch.Tensor] = ckpt['ema'].float().state_dict()
            ema.ema.load_state_dict(saved_ema)

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs
        
        del ckpt, csd

    # 图像尺寸
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # DP 模式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    
    # 训练
    for epoch in range(start_epoch, epochs):

        # Trainloader
        train_loader, dataset = create_mixed_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                                hyp=hyp, augment=True, rect=opt.rect, rank=RANK,
                                                workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                                prefix=colorstr('train: '), seed=2+RANK+epoch)
        mlc = int(np.concatenate(dataset.det_labels, 0)[:, 0].max())  # max label class 标签中共有多少类
        nb = len(train_loader)  # batch 总数
        assert mlc < de_nc, f'标签类别数 {mlc} 超过 {opt.data} 中的 nc={de_nc}. nc 值的范围是 0-{de_nc - 1}'

        # Process 0
        if RANK in [-1, 0]:
            val_loader = create_mixed_dataloader(val_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                                 hyp=hyp, rect=True, rank=-1,
                                                 workers=workers, pad=0.5,
                                                 prefix=colorstr('val: '))[0]
            if not resume:
                det_labels = np.concatenate(dataset.det_labels, 0)
                if plots:
                    plot_labels(det_labels, de_names, save_dir)
                
                # Anchors
                if not opt.noautoanchor:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                model.half().float()  # pre-reduce anchor precision
            
            callbacks.run('on_pretrain_routine_end')
        
        # DDP mode
        if cuda and RANK != -1:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        
        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= de_nc / 80. * 3. / nl


        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        for i, (imgs, det_labels, seg_labels, paths, _) in pbar:  # imgs.shape == torch.Size([16, 3, 640, 640])
            out_imgs = imgs.permute(0, 2, 3, 1).numpy()
            det_dict = {}
            for det in det_labels:
                key = det[0].item()
                if key in det_dict:
                    det_dict[key].append(det.numpy())
                else:
                    det_dict[key] = [det.numpy()]
            out_dets = det_dict.values()
            out_segs = seg_labels.numpy()
            for j, (out_img, out_det, out_seg, path) in enumerate(zip(out_imgs, out_dets, out_segs, paths)):
                h, w = out_img.shape[:2]
                out_img = np.ascontiguousarray(out_img)
                out_det = np.stack(out_det)
                out_det[:, 2:] = xywhn2xyxy(out_det[:, 2:], w, h)
                for det in out_det:
                    c = int(det[1])
                    label = de_names[c]
                    plot_one_box(det[2:], out_img, label=label, color=colors(c, True), line_thickness=1)
                # path_preffix = Path(path).stem
                path_preffix = f'{epoch}_{i}_{j}_'
                cv2.imwrite(f'runs/tmp{epoch}/{path_preffix}.jpg', cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
                file_name = f'runs/tmp{epoch}/{path_preffix}_label.png'
                if (out_seg > 0).any() and out_seg.min() >= 0 and out_seg.max() < 255:
                    lbl_pil = Image.fromarray(out_seg.astype(np.uint8), mode="P")
                    colormap = imgviz.label_colormap()
                    lbl_pil.putpalette(colormap.flatten())
                    lbl_pil.save(file_name)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='初始化网络权重文件的路径')
    parser.add_argument('--cfg', type=str, default='', help='网络模型配置文件 model.yaml 的位置')
    parser.add_argument('--data', type=str, default='core/data/blind.yaml', help='data.yaml 路径')
    parser.add_argument('--hyp', type=str, default='core/hyp/scratch.yaml', help='超参数路径')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='所有GPU的总batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复最近正在进行的训练')
    parser.add_argument('--nosave', action='store_true', help='只保存 final checkpoint')
    parser.add_argument('--noval', action='store_true', help='只验证 final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='禁用 autoanchor 检查')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda 设备, 例如 0 或者 0,1,2,3 或者 cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='将多类当作一类来训练')
    parser.add_argument('--adam', action='store_true', help='使用 torch.optim.Adam() 优化器')
    parser.add_argument('--sync-bn', action='store_true', help='使用 SyncBatchNorm, 仅 DDP 模式可用')
    parser.add_argument('--workers', type=int, default=8, help='dataloader workers 的最大值')
    parser.add_argument('--project', default='runs/train', help='保存到 project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='保存到 project/name')
    parser.add_argument('--exist-ok', action='store_true', help='project/name 可以存在, 不会新建')
    parser.add_argument('--quad', action='store_true', help='四分之一 dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑的 epsilon 值')
    parser.add_argument('--upload_dataset', action='store_true', help='数据集上传到 W&B 工件表')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='为 W&B 设置 bounding-box 图片打印间隔')
    parser.add_argument('--save_period', type=int, default=-1, help='每 "save_period" 个 epoch 之后保存一次 model')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='要使用的数据集工件的版本')
    parser.add_argument('--freeze', type=int, default=0, help='要冻结的 layers 数. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # 检查
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr('train: ' + ', '.join(f'{k}={v}' for k, v in vars(opt).items())))
        check_git_status()
        check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])
    
    # 恢复，回到之前位置
    if opt.resume and not check_wandb_resume(opt):  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'正在从 {ckpt} 恢复训练...')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # 检查文件
        assert len(opt.cfg) or len(opt.weights), '--cfg 和 --weights 必须指定'
        # TODO 下面一行可能要删
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # 扩展成 2 项 (train, test)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP 模式
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, '用于 DDP 的 CUDA 设备不足'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size 必须是 GPU 个数的倍数'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # 训练
    train(opt.hyp, opt, device, callbacks)
    if WORLD_SIZE > 1 and RANK == 0:
        _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]


def run(**kwargs):
    # 使用方法：import train; train.run(data='blind.yaml', imgsz=640, weights='yolov5s.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
