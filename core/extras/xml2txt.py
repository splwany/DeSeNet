import os
import xml.etree.ElementTree as ET

dirpath = 'datasets/blind/DeLabels_xml/train'  # 原来存放xml文件的目录
newdir = 'datasets/blind/DeLabels/train'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)

dict_info = {  # 有几个 属性 填写几个 label names
    'obstacle': 0,
    'pole': 1,
    'tree': 2,
    'automobile': 3,
    'motorcycle': 4,
    'bicycle': 5
}

for fp in os.listdir(dirpath):
    print(f'正在转换文件{fp}')
    if fp.endswith('.xml'):
        root = ET.parse(os.path.join(dirpath, fp)).getroot()

        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        sz = root.find('size')
        width = float(sz[0].text)
        height = float(sz[1].text)
        filename = root.find('filename').text
        out_str = ''
        for child in root.findall('object'):  # 找到图片中的所有框

            sub = child.find('bndbox')  # 找到框的标注值并进行读取
            label = child.find('name').text
            if label not in dict_info:
                continue
            label_ = dict_info[label]
            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
                x_center = (xmin + xmax) / (2 * width)
                x_center = '%.6f' % x_center
                y_center = (ymin + ymax) / (2 * height)
                y_center = '%.6f' % y_center
                w = (xmax - xmin) / width
                w = '%.6f' % w
                h = (ymax - ymin) / height
                h = '%.6f' % h
            except ZeroDivisionError:
                print(filename, '的 width有问题')
            out_str += ' '.join([str(label_), str(x_center), str(y_center), str(w), str(h) + '\n'])
        if out_str == '':
            continue
        with open(os.path.join(newdir, fp.split('.xml')[0] + '.txt'), 'w+') as f:
            f.write(out_str)
print('ok')