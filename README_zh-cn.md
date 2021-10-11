[English](https://github.com/splwany/DeSeNet/blob/main/README.md)
|
[简体中文](https://github.com/splwany/DeSeNet/blob/main/README_zh-cn.md)

# DeSeNet
实时目标检测和语义分割的端到端网络

## 数据集

解压到项目根目录的 `datasets` 文件夹中

[blind数据集（访问码：bc1d）](https://cloud.189.cn/web/share?code=zi2iAzbUvequ)

## DeSeNet 网络结构

### 1. YOLOv5s 网络结构

![preview](https://pic1.zhimg.com/v2-15e53f82f68e62ce1ea9a565121e21f8_r.jpg)

**注：**`c1`为输入通道数input_channel，`c2`为输出通道数output_channel，`k` 为卷积核大小kernel_size，`s` 为卷积核步长stride，`p` 为卷积边框padding，`g` 为卷积分组数groups，`act` 为是否使用激活函数，`e` 为瓶颈模块中隐藏层的扩张比例expansion

**注：**`batch` 为批大小batch_size， `channel` 为通道数，`width` 为图片的宽，`height`为图片的高

#### 1.1 Focus(c1=3, c2=32, k=32, s=1, p=None, g=1, act=True)

输入的形状Shape为：(batch, channel, width, height)，其中 `channel`为`c1`(即RGB三个通道)，其他根据参数决定；

将一张图片的像素等距离拆分成四张图，每张图的 `width`和`height`砍半，再在通道维度上拼接，故通道数为原来四倍，变成一个新的矩阵；

内部有一个 `conv`层（`Conv`模块）

将这个新矩阵输入 `conv` 层，并返回 `conv` 层的输出。

#### 1.2 Conv(c1, c2, k=1, s=1, p=None, g=1, act=True)

对应图中的 `CBL`；

内部的`ap`是根据`k`和`p`通过`autopad`方法计算得出的，`autopad`方法中，当`p`已知就直接返回`p`，当`p`未知，就返回`k`的一半并向下取整的值；

内部有一个 `conv`层（`nn.Conv2d`模块）

内部有一个 `bn`层（`nn.BatchNorm2d`模块）

内部有一个 `act`层（`nn.SiLU`激活模块，或`act`参数指定的激活模块）

输入矩阵依次通过 `conv`、`bn`、`act`层并将结果返回。

#### 1.3 C3(c1, c2, n=1, shortcut=True, g=1, e=0.5)

当`shortcut=True`时，对应图中`CSP1`；

当`shortcut=False`时，对应图中`CSP2`；

`n`对应图中`CSP1_x`中的`x`，其中`Bottleneck`模块的`e`为1；

#### 1.4 SPP(c1, c2, k=(5, 9, 13)) 空间金字塔池化层

隐藏层通道数为`c1`的一半；

内部有两个`Conv`模块，分别叫`cv1`和`cv2`;

内部有有三个`nn.MaxPoll2d`模块；

如图中`SPP`所示，输入的矩阵先经过`cv1`，再依次经过`k`为5、9、13的池化层得到三个结果，与`cv1`的输出一起在通道维上拼接，故通道数变为输入的四倍，作为输入矩阵进入`cv2`，得到一个通道数为`c2`的输出，并返回。

#### 1.5 Concat(dimension=1) 拼接层

只是利用`torch.cat()`方法，将输入矩阵在`dimension`维度上拼接，并返回结果。

#### 1.6 Detect(nc=80, anchor=(), ch=(), inplace=True) YOLO的检测头

`nc`为数据集类别数；`no = nc + 5`，即每个`anchor`的输出个数，包括`nc`个类别和目标位置框中心点的坐标和宽高，以及置信度；

### 2 Segmentation 模块

#### 2.1 SegMaskPSP(n_segcls=2, n=1, c_hide=256, shortcut=False, ch=()) 语义分割检测头

