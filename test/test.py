# import yaml
# import argparse
from hashlib import pbkdf2_hmac
from logging import setLoggerClass
from pathlib import Path

# from torch.utils import data
import os

import torch

import numpy as np
from PIL import Image
import imgviz
import random
import cv2

import sys
root_path = str(Path(__file__).parent.parent.absolute())
sys.path.append(root_path)

from core.utils.general import generate_seg_labels_img


# indices = range(10)
# print([10] + list(np.random.choice(indices, size=3, replace=False)))
# print([10] + random.sample(indices, k=3))
img_path = 'datasets/blind/images/train/blind_44_12.jpg'
img = cv2.imread(img_path)
h0, w0 = img.shape[:2]
r = 640 / max(h0, w0)
if r != 1:
    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_CUBIC)