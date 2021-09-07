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
import time

import sys
root_path = str(Path(__file__).parent.parent.absolute())
sys.path.append(root_path)

from core.utils.general import generate_seg_labels_img
from core.utils.mixed_datasets import exif_size, correct_rotation


yv, xv = torch.meshgrid([torch.arange(4), torch.arange(4)])
print(yv)
print(xv)
print(torch.stack((xv, yv), 2).view(1, 1, 4, 4, 2))