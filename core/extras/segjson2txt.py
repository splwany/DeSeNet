import os
import json
import itertools
import numpy as np


dirpath = 'datasets/blind/SeLabels_json/train'  # 原来存放xml文件的目录
newdir = 'datasets/blind/SeLabels/train'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)

dict_info = {  # 有几个 属性 填写几个 label names
    '_background_': 0,
    'step': 1,
    'shoulder': 2
}

for fp in os.listdir(dirpath):
    if fp.endswith('.json'):
        print(f'正在转换文件{fp}')
        with open(os.path.join(dirpath, fp), 'r') as json_file:
            out_str = ''
            json_datas = json.load(json_file)
            shapes = json_datas['shapes']
            for shape in shapes:
                label = shape['label']
                if label in dict_info:
                    label = dict_info[label]
                    points = shape['points']
                    img_size = (json_datas['imageWidth'], json_datas['imageHeight'])
                    if len(points):
                        points = np.array(points, dtype=np.float64)
                        points = points / np.array(img_size, dtype=np.float64)
                        points = np.clip(points, 0, 1)
                        points = ' '.join(map(str, itertools.chain(*points)))
                        out_str += f'{label} {points}\n'
            if out_str != '':
                with open(os.path.join(newdir, fp.split('.json')[0] + '.txt'), 'w+') as f:
                    f.write(out_str)
print('ok')