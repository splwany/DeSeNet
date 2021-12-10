import argparse
import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
CORE = ROOT / 'core'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to path
if str(CORE) not in sys.path:
    sys.path.append(str(CORE))  # add CORE to path
ROOT = ROOT.relative_to(Path.cwd())  # relative

from core.models.experimental import attempt_load
from core.models.yolo import Model
from core.utils.autoanchor import check_anchors
from core.utils.callbacks import Callbacks
from core.utils.general import (check_dataset, check_file, check_git_status,
                                check_img_size, check_requirements,
                                check_suffix, check_yaml, colorstr,
                                get_latest_run, increment_path, init_seeds,
                                labels_to_class_weights,
                                labels_to_image_weights, methods, one_cycle,
                                print_args, set_logging, strip_optimizer,
                                xywhn2xyxy, xyxy2xywh)
from core.utils.google_utils import attempt_download
from core.utils.loggers import Loggers
from core.utils.loggers.wandb.wandb_utils import check_wandb_resume
from core.utils.loss import ComputeLoss, SegmentationLosses
from core.utils.metrics import fitness_det_seg
from core.utils.mixed_datasets import (InfiniteDataLoader,
                                       create_mixed_dataloader)
from core.utils.plots import colors, plot_images, plot_labels, plot_lr_scheduler
from core.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel,
                                    intersect_dicts, select_device,
                                    torch_distributed_zero_first)

import scripts.val as val  # for end-of-epoch mAP

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp, opt, device: torch.device, callbacks):
    # 从 opt 中获取相关参数
    save_dir, epochs, batch_size, weights, single_cls = Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls
    data, cfg, resume, noval, nosave, workers, freeze = opt.data, opt.cfg, opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # 目录
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # 超参数
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # 加载 hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # 保存运行的设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp
        
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
        # TODO 修改模型参数，修改模型结构
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
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
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
            ema.updates = ckpt['updates']

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

    # Trainloader
    train_loader, dataset = create_mixed_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                                    hyp=hyp, augment=True, rect=opt.rect, rank=RANK,
                                                    workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                                    prefix=colorstr('train: '))
    # assert isinstance(train_loader, InfiniteDataLoader)
    mlc = int(np.concatenate(dataset.det_labels, 0)[:, 0].max())  # max label class 标签中共有多少类
    nb = len(train_loader)  # batch 总数
    assert mlc < de_nc, f'目标检测标签类别数 {mlc} 超过 {opt.data} 中的 nc={de_nc}. nc 值的范围是 0{"-" + str(de_nc - 1) if de_nc > 1 else ""}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_mixed_dataloader(val_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                             hyp=hyp, rect=True, rank=-1,
                                             workers=workers, pad=0.5,
                                             prefix=colorstr('val: '))[0]
        if not resume:
            det_labels = np.concatenate(dataset.det_labels, 0)
            seg_labels = dataset.seg_labels
            if plots:
                plot_labels(det_labels, de_names, save_dir)
                # TODO 实现语义分割 label 的显示
            
            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            # pre-reduce anchor precision
            model.half().float()
        
        callbacks.run('on_pretrain_routine_end')
    
    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= de_nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.de_nc = de_nc  # attach number of det_classes to model
    model.se_nc = se_nc  # attach number of seg_classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.det_labels, de_nc).to(device) * de_nc 
    model.de_names = de_names
    model.se_names = se_names
    
    # 开始训练
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # TODO 还需添加 se_nc 的 mAP
    maps = np.zeros(de_nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5~.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # 不要移除这一行
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class

    # Base, PSP 和 Lab 用这个，无 aux
    compute_seg_loss = SegmentationLosses()

    detgain, seggain = 0.15, 0.85  # 目标检测、语义分隔 比例
    # CE、1/8单输入、batchsize13用0.65,0.35左右,注意64向下取整的梯度积累，比13*4=52大(12*5=64)通常应该降低分割损失比例或调小学习率

    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f'Logging results to {colorstr("bold", save_dir)}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch -----------------------------------------------------------
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / de_nc  # class weights
            iw = labels_to_image_weights(dataset.det_labels, nc=de_nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        if opt.rect:
            indices = dataset.indices
            pad = (batch_size // WORLD_SIZE) - (len(indices) % (batch_size // WORLD_SIZE))
            if pad > 0:
                indices += random.choices(indices[(pad - (batch_size // WORLD_SIZE)):], k=pad)
            indices = np.asarray(indices).reshape(-1, (batch_size // WORLD_SIZE))
            np.random.shuffle(indices)
            dataset.indices = list(indices.flatten())
        else:
            random.shuffle(dataset.indices)

        mloss = torch.zeros(3, device=device)  # 目标检测 mean losses
        msegloss = torch.zeros(1, device=device)  # 混合的 mean losses, 两者计算也可知分割 loss
        if RANK != -1:
            assert isinstance(train_loader.sampler, torch_data.distributed.DistributedSampler)
            train_loader.sampler.set_epoch(epoch)  # shuffle时，保证每个epoch的划分不同
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_men', 'box', 'obj', 'cls', 'seg', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # 进度条
        optimizer.zero_grad()
        for i, (imgs, det_labels, seg_labels, paths, _) in pbar:  # imgs.shape == torch.Size([batch, 3, 640, 640])
            # TODO 以下两行代码为测试数据集加载效果的，这些最后需要删掉
            # path_prefix = f'{epoch}_{i}'
            # plot_images(imgs, det_labels, seg_labels, paths, f'runs/tmp{epoch}/{path_prefix}.jpg', f'runs/tmp{epoch}/{path_prefix}_label.jpg', de_names, se_names)

            ni = i + nb * epoch  # number integrated batches (since train start)
            assert isinstance(imgs, torch.Tensor)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward 此处有修改，否则batchsize只能取单检测时候的一半，这种写法可以更大一点
            with amp.autocast(enabled=cuda):  # 混合精度训练中用来代替autograd
                det_pred, seg_pred = model(imgs)  # forward
                # TODO 更改 loss 中的参数
                det_loss, det_loss_items = compute_loss(det_pred, det_labels.to(device))  # loss scaled by batch_size
                seg_loss = compute_seg_loss(seg_pred, seg_labels.to(device))
                if RANK != -1:
                    det_loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    seg_loss *= WORLD_SIZE  # TODO 是否是乘 batch_size
                if opt.quad:
                    det_loss *= 4.
                    seg_loss *= 4.
                det_loss *= detgain  # 目标检测的比例
                seg_loss *= seggain  # 语义分割的比例

            # Backward
            scaler.scale(det_loss).backward(retain_graph=True)
            scaler.scale(seg_loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:  # 梯度积累accumulate次后才优化
                scaler.step(optimizer)  # optimizer.step 混合精度训练优化时用scaler
                scaler.update()
                optimizer.zero_grad()  # 每次更新完参数才清空梯度，不更新时累计
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + det_loss_items) / (i + 1)  # update mean losses
                msegloss = (msegloss * i + seg_loss.detach() / batch_size) / (i + 1)  # update mean seglosses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, msegloss, det_labels.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, det_labels, seg_labels, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()  # 更新 Scheduler

        # DDP process 0 or single-GPU
        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

            # TODO pixACC, mIoU
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # 计算 mAP

                mIoU = val.seg_validation(model=ema.ema,
                                          n_segcls=3,
                                          valloader=val_loader,
                                          half_precision=True)

                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)

            # Update best mAP
            fi = fitness_det_seg(np.array(results).reshape(1, -1), mIoU)  # weighted combination of [P, R, mAP@.5, mAP@.5-.95] 按0.1*AP.5+0.9*AP.5:.95指标衡量模型
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in (last, best):
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            iou_thres=0.7,  # best pycocotools results at 0.65
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=False,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss,)  # val best model with plots

        callbacks.run('on_train_end', last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='初始化网络权重文件的路径')
    parser.add_argument('--cfg', type=str, default=ROOT / 'core/models/yolov5s_seg.yaml', help='网络模型配置文件 model.yaml 的位置')
    parser.add_argument('--data', type=str, default=ROOT / 'core/data/blind.yaml', help='data.yaml 路径')
    parser.add_argument('--hyp', type=str, default=ROOT / 'core/hyp/scratch.yaml', help='超参数路径')
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
    parser.add_argument('--project', default=ROOT / 'runs/train', help='保存到 project/name')
    parser.add_argument('--name', default='exp', help='保存到 project/name')
    parser.add_argument('--exist-ok', action='store_true', help='project/name 可以存在, 不会新建')
    parser.add_argument('--quad', action='store_true', help='四分之一 dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑的 epsilon 值')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='要冻结的 layers 数. backbone=10, all=24')
    parser.add_argument('--save_period', type=int, default=-1, help='每 "save_period" 个 epoch 之后保存一次 model')

    # Weights & Biases arguments
    parser.add_argument('--entity', default='splwany', help='W&B entity')
    parser.add_argument('--upload_dataset', action='store_true', help='数据集上传到 W&B 工件表')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='为 W&B 设置 bounding-box 图片打印间隔')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='要使用的数据集工件的版本')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # 检查
    set_logging(RANK)
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])
    
    # 恢复，回到之前位置
    if opt.resume and not check_wandb_resume(opt):  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'正在从 {ckpt} 恢复训练...')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # 检查文件
        assert len(opt.cfg) or len(opt.weights), '--cfg 和 --weights 必须指定'
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
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


def run(**kwargs):
    # 使用方法：import train; train.run(data='blind.yaml', imgsz=640, weights='yolov5s.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
