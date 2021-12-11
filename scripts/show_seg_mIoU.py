import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

import torch.nn.functional as F
from core.models.experimental import attempt_load
from core.utils.callbacks import Callbacks
from core.utils.mixed_datasets import create_mixed_dataloader
from core.utils.general import (box_iou, check_dataset, check_img_size,
                                check_requirements, check_suffix, check_yaml,
                                coco80_to_coco91_class, colorstr,
                                increment_path, non_max_suppression,
                                print_args, scale_coords, set_logging,
                                xywh2xyxy, xyxy2xywh)
from core.utils.metrics import (ConfusionMatrix, ap_per_class,
                                batch_intersection_union, batch_pix_accuracy)  # 后两个新增分割
from core.utils.plots import output_to_target, segoutput_to_target, plot_images, plot_val_study
from core.utils.torch_utils import select_device, time_sync
from scripts.val import seg_validation


@torch.no_grad()
def run(data,
        hyp,
        weights,
        batch_size=32,
        imgsz=640,
        device='',
        single_cls=False,
        half=True
        ):
    device = select_device(device, batch_size=batch_size)
    model = attempt_load(weights, map_location=device)
    gs = max(int(model.stride.max()), 32)
    data = check_dataset(data)
    val_loader = create_mixed_dataloader(data['val'], imgsz, batch_size, gs, single_cls, hyp,
                                             rect=True, rank=-1,
                                             pad=0.5,
                                             prefix=colorstr('val: '))[0]
    seg_validation(model, 3, val_loader, half)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--data', type=str, default=ROOT / 'core/data/blind.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'core/hyp/scratch.yaml', help='超参数路径')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt


def main(opt):
    set_logging()
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)