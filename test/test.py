# import yaml
# import argparse
import argparse
from hashlib import pbkdf2_hmac
from logging import addLevelName, setLoggerClass
from pathlib import Path

# from torch.utils import data
import os

import torch
import pandas as pd

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
from core.utils.mixed_datasets import exif_size, exif_transpose

from itertools import repeat
from torch.nn import BCELoss, BCEWithLogitsLoss, Sigmoid


save_dir = Path('runs/train/first/')
files = sorted(save_dir.glob('train*[jpg,png]'))
print([str(f) for f in files if f.exists()])