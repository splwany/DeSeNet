# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
from pathlib import Path
import warnings
from threading import Thread
from logging import Logger

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter

from core.utils.general import colorstr, emojis
from core.utils.loggers.wandb.wandb_utils import WandbLogger
from core.utils.plots import plot_images, plot_results
from core.utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb', 'wandb')  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv('RANK', -1))

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    # if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in [-1, 0]:
    #     wandb_login_success = wandb.login(timeout=30)
    #     if not wandb_login_success:
    #         wandb = None
except (ImportError, AssertionError):
    wandb = None


class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                     'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                     'x/lr0', 'x/lr1', 'x/lr2']  # params
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            print(emojis(s))

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include:
            assert isinstance(s, Path), f's 必须为 Path 类型路径, 当前为 {type(s)}'
            assert isinstance(self.logger, Logger)
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and 'wandb' in self.include:
            assert self.opt is not None
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        assert isinstance(self.save_dir, Path)
        paths = self.save_dir.glob('*labels*.jpg')  # training labels
        if self.wandb:
            assert wandb is not None
            self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})

    def on_train_batch_end(self, ni, model, imgs, targets, seg_targets, paths, plots, sync_bn):
        # Callback runs on train batch end
        if plots:
            de_names = model.module.de_names if hasattr(model, 'module') else model.de_names
            se_names = model.module.se_names if hasattr(model, 'module') else model.se_names
            if ni == 0:
                if not sync_bn:  # tb.add_graph() --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # suppress jit trace warning
                        self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
            assert isinstance(self.save_dir, Path)
            assert wandb is not None
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                seg_f = self.save_dir / f'train_batch{ni}_seg.png'  # segfilename
                Thread(target=plot_images, args=(imgs, targets, seg_targets, paths, f, seg_f, de_names, se_names), daemon=True).start()
            if self.wandb and ni == 5:
                files = sorted(self.save_dir.glob('train*[jpg,png]'))
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, seg_pred, path, de_names, se_names, im):
        # Callback runs on val image end
        if self.wandb:
            self.wandb.val_one_image(pred, predn, seg_pred, path, de_names, se_names, im)

    def on_val_end(self):
        # Callback runs on val end
        if self.wandb:
            assert isinstance(self.save_dir, Path)
            assert wandb is not None
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv:
            assert isinstance(self.save_dir, Path)
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            assert self.opt is not None
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, plots, epoch):
        # Callback runs on training end
        assert isinstance(self.save_dir, Path)
        if plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter

        if self.tb:
            import cv2
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

        if self.wandb:
            assert wandb is not None
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            # if not self.opt.evolve:
            wandb.log_artifact(str(best if best.exists() else last), type='model',
                                name='run_' + self.wandb.wandb_run.id + '_model',
                                aliases=['latest', 'best', 'stripped'])
            self.wandb.finish_run()
            # else:
            #     self.wandb.finish_run()
            #     self.wandb = WandbLogger(self.opt)