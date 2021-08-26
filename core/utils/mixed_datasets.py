import enum
import glob
import hashlib
import logging
import math
import os
import random
import shutil
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from sys import is_finalizing
from tarfile import is_tarfile
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .general import (check_requirements, clean_str, resample_segments,
                      segment2box, segments2boxes, xyn2xy, xywh2xyxy,
                      xywhn2xyxy, xyxy2xywh, seg_xyn2xy, generate_seg_labels_img)
from .torch_utils import torch_distributed_zero_first


# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # 可用的图片后缀名
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # 可用的视频后缀名
logger = logging.getLogger(__name__)

# 获取exif中Orientation标签key值
orientation = next(filter(lambda item: item[1] == 'Orientation', ExifTags.TAGS.items()))[0]


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = 0
    for p in paths:
        p = Path(p)
        if p.is_file():
            size += p.stat().st_size
        elif p.is_dir():
            size += sum([t.stat().st_size for t in p.glob('*.*')])

    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # 返回exif修正后的PIL图片宽高 (weight, height)
    s = img.size  # (width, height)
    try:
        rotation = img.getexif()[orientation]
        if rotation in [Image.ROTATE_90, Image.ROTATE_270]:
            s = (s[1], s[0])
    except:
        pass

    return s


def correct_rotation(img):
    rotation_dict = {
        8: Image.ROTATE_90,
        3: Image.ROTATE_180,
        6: Image.ROTATE_270
    }
    try:
        rotation = img.getexif()[orientation]
        if rotation in rotation_dict:
            return img.transpose(rotation_dict[rotation])
    except:
        pass
    return img


def create_mixed_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                            rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', seed=0):
    # 确保 DDP 中的主进程先加载 dataset，这样其他进程可以使用其缓存
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      seed=seed)
    
    batch_size = min(batch_size, len(dataset))
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1
    nw = min([cpu_count // world_size, batch_size if batch_size > 1 else 0, workers])  # workers 数量
    sampler = torch_data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch_data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch_data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def img2label_paths(img_paths):
    de_labels = []  # 目标检测的 label_paths
    se_labels = []  # 语义分割的 label_paths
    for x in img_paths:
        x = Path(x)
        f_name = x.with_suffix('.txt').name
        de_parent = x.parent.parent.with_name('DeLabels') / x.parent.name
        de_labels.append(str(de_parent / f_name))
        se_parent = x.parent.parent.with_name('SeLabels') / x.parent.name
        se_labels.append(str(se_parent / f_name))

    return de_labels, se_labels


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', seed=0):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # 同时加载四张图片为一张 mosaic 图（仅在训练期间有效）
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path

        p = Path()

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():  # dir
                    f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with p.open('r') as t:
                        t = t.read().strip().splitlines()
                        f += [p.parent / x.lstrip(os.sep) for x in t]  # lacal to global path
                else:
                    raise Exception(f'{prefix}{p} 不存在')
            self.img_files = sorted([str(x) for x in f if x.suffix[1:].lower() in img_formats])
            assert self.img_files, f'{prefix}没找到图片'
        except Exception as e:
            raise Exception(f'{prefix}数据加载错误，{path}: {e}')
        
        # 检查缓存
        self.de_label_files, self.se_label_files = img2label_paths(self.img_files)
        cache_parent = p.parent.parent if p.is_file() else Path(self.de_label_files[0]).parent.parent.parent
        cache_name = (p if p.is_file() else Path(self.de_label_files[0]).parent).with_suffix('.cache').name
        cache_path = cache_parent / cache_name
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True
            if cache['hash'] != get_hash(self.de_label_files + self.se_label_files + self.img_files):  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache
        
        # 显示缓存文件
        nf, nm, ne, nc, nu = cache.pop('results')  # found, missing, empty, corrupted, used
        if exists:
            d = f"已扫描 '{cache_path}' 中的图片和标注... 发现{nf}个，丢失{nm}个，空{ne}个，损坏{nc}个，使用{nu}个"
            tqdm(None, desc=prefix + d, total=nu, initial=nu)  # 显示 cache 结果
        assert nu > 0 or not augment, f'{prefix}{cache_path}中无标注，无法训练'

        # 读取缓存文件
        cache.pop('hash')  # 移除 hash
        cache.pop('version')  # 移除 version
        cache_items = list(cache.items())
        random.seed(seed)
        random.shuffle(cache_items)
        self.img_files = [item[0] for item in cache_items]  # update
        cache_values = [item[1] for item in cache_items]
        self.shapes, self.det_labels, self.seg_labels = zip(*cache_values)
        self.shapes = np.array(self.shapes, dtype=np.float64)
        self.de_label_files, self.se_label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for d, s in zip(self.det_labels, self.seg_labels):
                d[:, 0] = 0
                s[:, 0] = 0
        
        bi = np.floor(np.arange(nu) / batch_size).astype(np.int32)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = nu
        self.indices = range(nu)

        # 矩形训练        
        if self.rect:
            # 按长宽比排序
            ar = self.shapes[:, 1] / self.shapes[:, 0]  # aspect ratio (长宽比)
            irect = ar.argsort()
            ar = ar[irect]
            # 图片、标注 按图片长宽比从小到大排序
            self.img_files = [self.img_files[i] for i in irect]  # 图片路径
            self.det_labels = [self.det_labels[i] for i in irect]  # 目标检测标注内容
            self.seg_labels = [self.seg_labels[i] for i in irect]  # 语义分割标注内容
            self.shapes = self.shapes[irect]  # 图片尺寸

            # 设置用来训练的图片的尺寸
            shapes = []
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes.append([maxi, 1])
                elif mini > 1:
                    shapes.append([1, 1 / mini])
                else:
                    shapes.append([1, 1])
            
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int32) * stride
    
        # TODO 将图片缓存到内存以加速训练（注意：大数据集可能超过RAM）

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # 缓存数据集标注，检查图片并读取形状
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupted
        pbar = tqdm(zip(self.img_files, self.de_label_files, self.se_label_files), desc='正在扫描图片', total=len(self.img_files))
        for im_file, dlb_file, slb_file in pbar:
            try:
                # 验证图片
                with Image.open(im_file) as im:
                    im.verify()  # PIL verify
                    shape = exif_size(im)  # image size
                    assert im.format.lower() in img_formats, f'不支持的图片格式：{im.format}'
                    assert (shape[0] > 9) and (shape[1] > 9), f'图片尺寸不能小于10像素，当前：{shape}'

                # 验证目标检测标注
                det_labels = np.zeros((0, 5), dtype=np.float64)
                if os.path.isfile(dlb_file):
                    nf += 1  # 目标检测标签已找到
                    with open(dlb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if len(l):
                        det_labels = np.array(l, dtype=np.float64)
                        assert det_labels.shape[1] == 5, '标注要求每一条有五个值'
                        assert (det_labels >= 0).all(), '存在类别或坐标为负值的标注'
                        assert (det_labels[:, 1:] <= 1).all(), '未归一化或超出坐标限制'
                        assert np.unique(det_labels, axis=0).shape[0] == det_labels.shape[0], '存在重复标注'
                    else:
                        ne += 1  # 目标检测标签为空
                else:
                    nm += 1

                # 验证语义分割标注
                seg_labels = np.zeros((0, 2))
                if os.path.isfile(slb_file):
                    with open(slb_file, 'r') as f:
                        l = []
                        for line in f.read().strip().splitlines():
                            items = line.split()
                            l.append(np.array(items, dtype=np.float64))
                    if len(l):
                        assert all([(item >= 0).all() for item in l]), '存在类别或坐标为负值的标注'
                        assert all([(item[1:] <= 1).all() for item in l]), '未归一化或超出坐标限制'
                        seg_labels = np.array([[int(item[0]), np.array(item[1:], dtype=np.float64)] for item in l], dtype=object)
                        seg_labels[:, 1] = [item.reshape(-1, 2) for item in seg_labels[:, 1]]

                if len(det_labels) or len(seg_labels):
                    x[im_file] = [shape, det_labels, seg_labels]

            except Exception as e:
                nc += 1
                logging.info(f'{prefix}警告：正在忽略损坏的图片与标注 {im_file}: {e}')
            
            pbar.desc = f"{prefix}正在扫描 '{path.parent / path.stem}' 中的图片和标注" \
                        f"发现{nf}个，丢失{nm}个，空{ne}个，损坏{nc}个"
        pbar.close()

        if nf == 0:
            logging.info(f'{prefix}警告：{path}中没找到标注')
        
        nu = len(x)  # number used
        x['hash'] = get_hash(self.de_label_files + self.se_label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, nu
        x['version'] = 0.2  # cache version
        try:
            torch.save(x, path)  # save cache for next time
            logging.info(f'{prefix}新的cache已创建：{path}')
        except Exception as e:
            logging.info(f'{prefix}警告：cache所在目录 {path.parent} 不可写：{e}')  # path 不可写
        return x
    
    def __len__(self):
        return self.n

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        assert hyp is not None
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # load mosaic
            img, det_labels, seg_labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            # if random.random() < hyp['mixup']:
            #     img2, det_labels2, seg_labels2 = load_mosaic(self, random.randint(0, self.n - 1))
            #     r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
            #     img = (img * r + img2 * (1 - r)).astype(np.uint8)
            #     det_labels = np.concatenate((det_labels, det_labels2), 0)
            #     seg_labels = np.concatenate((seg_labels, seg_labels2), 0)
        
        else:
            # load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # 最终的 letterboxed 尺寸
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            det_labels = self.det_labels[index].copy()
            seg_labels = self.seg_labels[index].copy()
            assert isinstance(det_labels, np.ndarray), 'det_labels 应为 numpy 数组'
            assert isinstance(seg_labels, np.ndarray), 'seg_labels 应为 numpy 数组'
            if det_labels.size:  # normalized xywh to pixwl xyxy format
                det_labels[:, 1:] = xywhn2xyxy(det_labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            if seg_labels.size:
                seg_labels[:, 1] = seg_xyn2xy(seg_labels[:, 1], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        
        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, det_labels, seg_labels = random_perspective(img, det_labels, seg_labels,
                                                                 degrees=hyp['degrees'],
                                                                 translate=hyp['translate'],
                                                                 scale=hyp['scale'],
                                                                 shear=hyp['shear'],
                                                                 perspective=hyp['perspective'])
            
            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)
        
        det_labels_num = len(det_labels)  # number of det_labels
        if det_labels_num:
            det_labels[:, 1:5] = xyxy2xywh(det_labels[:, 1:5])  # convert xyxy to xywh
            det_labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            det_labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        seg_labels_num = len(seg_labels)  # number of seg_labels
        
        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if seg_labels_num:
                    for item in seg_labels[:, 1]:
                        item[:, 1] = img.shape[0] - item[:, 1]
                if det_labels_num:
                    det_labels[:, 2] = 1 - det_labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if seg_labels_num:
                    for item in seg_labels[:, 1]:
                        item[:, 0] = img.shape[1] - item[:, 0]
                if det_labels_num:
                    det_labels[:, 1] = 1 - det_labels[:, 1]

        zeros = torch.zeros(len(det_labels), 1)
        det_labels = torch.cat([zeros, torch.from_numpy(det_labels)], dim=1)
        
        # 生成语义分割图片
        seg_labels_img = generate_seg_labels_img(seg_labels, img.shape[:2])

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)  # 将内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        img = torch.from_numpy(img)

        return img, det_labels, seg_labels_img, self.img_files[index], shapes
    
    @staticmethod
    def collate_fn(batch):
        img, det_label, seg_label_img, path, shapes = zip(*batch)  #  transposed
        for i in range(len(det_label)):
            # add target image index for build_targets()
            det_label[i][:, 0] = i
        return torch.stack(img, 0), torch.cat(det_label, 0), torch.stack(seg_label_img, 0), path, shapes
    
    @staticmethod
    def collate_fn4(batch):
        assert len(batch) >= 4, 'batch size must not less than 4 when using collate_fn4'
        img, det_label, seg_label_img, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, det_label4, seg_label_img4, path4, shapes4 = [], [], [], path[::4], shapes[::4]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale

        dl = []
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[0].type(img[i].type())
                dl = det_label[i]
                sl = F.interpolate(seg_label_img[i].float().unsqueeze(0).unsqueeze(0), scale_factor=2., mode='area')[0].type(img[i].type()).squeeze().int()
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                dl = torch.cat((det_label[i], det_label[i + 1] + ho, det_label[i + 2] + wo, det_label[i + 3] + ho + wo), 0) * s
                sl = torch.cat((torch.cat((seg_label_img[i], seg_label_img[i + 1]), 0), torch.cat((seg_label_img[i + 2], seg_label_img[i + 3]), 0)), 1)
            img4.append(im)
            det_label4.append(dl)
            seg_label_img4.append(sl)

        for i in range(len(det_label4)):
            # add target image index for build_targets()
            det_label4[i][:, 0] = i
        return torch.stack(img4, 0), torch.cat(det_label4, 0), torch.stack(seg_label_img4, 0), path4, shapes4


# 辅助函数 ----------------------------------------------------------------
def load_image(self: LoadImagesAndLabels, index: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    # 从数据集加载一张图片和其对应的语义分割标注图片，返回 img, seg_label_img, original hw, resized hw
    assert isinstance(self.img_files, list)
    img_path = self.img_files[index]  # 图片路径
    
    assert os.path.isfile(img_path), f'图片未找到：{img_path}'
    img = Image.open(img_path)  # RGB
    img = correct_rotation(img)  # 图片旋转矫正
    w0, h0 = img.size  # 原始 wh
    assert img.size == tuple(self.shapes[index]), f'图片尺寸与缓存不符：{img_path}'
    r = self.img_size / max(w0, h0)  # 比例

    if r != 1:  # 如果尺寸不一致
        new_wh = (int(w0 * r), int(h0 * r))
        img = img.resize(new_wh, Image.ANTIALIAS)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img, (h0, w0), (img.shape[0], img.shape[1])


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_mosaic(self: LoadImagesAndLabels, index):
    # 加载四张图片拼成一张图
    det_labels4, seg_labels4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
    indices = [index] + random.sample(self.indices, k=3)  # 除了当前index外，再额外随机选三个，组成四个index
    img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    for i, index in enumerate(indices):
        # 加载图片
        img, _, (h, w) = load_image(self, index)

        # 将 img 放到 img4 中，将 seg_label_img 放到 seg_label_img4中
        x1a, y1a, x2a, y2a = 0, 0, 0, 0
        x1b, y1b, x2b, y2b = 0, 0, 0, 0
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # 目标检测标注
        det_labels = self.det_labels[index].copy()
        assert isinstance(det_labels, np.ndarray)
        if det_labels.size:
            det_labels[:, 1:] = xywhn2xyxy(det_labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        det_labels4.append(det_labels)

        # 语义分割标注
        seg_labels = self.seg_labels[index].copy()
        assert isinstance(seg_labels, np.ndarray)
        if seg_labels.size:
            seg_labels[:, 1] = seg_xyn2xy(seg_labels[:, 1], w, h, padw, padh)
        seg_labels4.append(seg_labels)
    
    # 拼接/修剪 标注
    det_labels4 = np.concatenate(det_labels4, 0)  # 将三维的 det_labels4 拼成二维的
    np.clip(det_labels4[:, 1:], 0, 2 * s, out=det_labels4[:, 1:])  # clip when using random_perspective()
    seg_labels4 = np.concatenate(seg_labels4, 0)  # 将三维的 seg_labels4 拼成二维的
    for x in seg_labels4:
        np.clip(x[1], 0, 2 * s, out=x[1])
    
    # 数据扩充
    assert self.hyp is not None, '没有定义可用的 hyp'
    img4, det_labels4, seg_labels4 = random_perspective(img4, det_labels4, seg_labels4,
                                                       degrees=self.hyp['degrees'],
                                                       translate=self.hyp['translate'],
                                                       scale=self.hyp['scale'],
                                                       shear=self.hyp['shear'],
                                                       perspective=self.hyp['perspective'],
                                                       border=self.mosaic_border)
    return img4, det_labels4, seg_labels4


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # 调整图片大小并设置内边距，同时满足多步幅的约束
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = (r, r)  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, det_targets=(), seg_targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=.0, border=(0, 0)):
    """随机视角"""
    
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # 中心
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    
    # Transform det_label coordinates
    n = len(det_targets)
    if n:
        new = np.zeros((n, 4))

        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = det_targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=det_targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        det_targets = det_targets[i]
        det_targets[:, 1:5] = new[i]
    
    # Transform seg_label coordinates
    n = len(seg_targets)
    if n:
        # warp boxes
        xy = seg_targets[:, 1]  # xy
        for i, item in enumerate(xy):
            ones = np.ones((len(item), 1))
            xy[i] = np.concatenate([item, ones], axis=1)
        # transform
        for i, item in enumerate(xy):
            xy[i] = item @ M.T
        # perspective rescale or affine
        for i, item in enumerate(xy):
            xy[i] = item[:, :2] / item[:, 2:3] if perspective else item[:, :2]

    return img, det_targets, seg_targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16, special_classes=0):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & ((ar < ar_thr) | ((special_classes == 0) and (ar < 120)))  # candidates
