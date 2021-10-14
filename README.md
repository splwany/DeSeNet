[English](https://github.com/splwany/DeSeNet/blob/main/README_en.md)
|
[简体中文](https://github.com/splwany/DeSeNet/blob/main/README.md)

# DeSeNet
实时目标检测和语义分割的端到端网络

## 数据集

解压到项目根目录的 `datasets` 文件夹中

[blind数据集（访问码：bc1d）](https://cloud.189.cn/web/share?code=zi2iAzbUvequ)

## DeSeNet 网络结构

```
               from  n    params  module                                  arguments
  0                -1  1      3520  core.models.common.Focus                [3, 32, 3]
  1                -1  1     18560  core.models.common.Conv                 [32, 64, 3, 2]
  2                -1  1     18816  core.models.common.C3                   [64, 64, 1]
  3                -1  1     73984  core.models.common.Conv                 [64, 128, 3, 2]
  4                -1  3    156928  core.models.common.C3                   [128, 128, 3]
  5                -1  1    295424  core.models.common.Conv                 [128, 256, 3, 2]
  6                -1  3    625152  core.models.common.C3                   [256, 256, 3]
  7                -1  1   1180672  core.models.common.Conv                 [256, 512, 3, 2]
  8                -1  1    656896  core.models.common.SPP                  [512, 512, [5, 9, 13]]        
  9                -1  1   1182720  core.models.common.C3                   [512, 512, 1, False]
 10                -1  1    131584  core.models.common.Conv                 [512, 256, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  core.models.common.Concat               [1]
 13                -1  1    361984  core.models.common.C3                   [512, 256, 1, False]
 14                -1  1     33024  core.models.common.Conv                 [256, 128, 1, 1]
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  core.models.common.Concat               [1]
 17                -1  1     90880  core.models.common.C3                   [256, 128, 1, False]
 18                -1  1    147712  core.models.common.Conv                 [128, 128, 3, 2]
 19          [-1, 14]  1         0  core.models.common.Concat               [1]
 20                -1  1    296448  core.models.common.C3                   [256, 256, 1, False]
 21                -1  1    590336  core.models.common.Conv                 [256, 256, 3, 2]
 22          [-1, 10]  1         0  core.models.common.Concat               [1]
 23                -1  1   1182720  core.models.common.C3                   [512, 512, 1, False]
 24      [16, 19, 22]  1    670082  core.models.yolo.SegMaskPSP             [2, 1, 128, False, [256, 256, 512]]
 25      [17, 20, 23]  1     29667  core.models.yolo.Detect                 [6, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
```

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

### 2. Segmentation 模块

![](https://github.com/splwany/DeSeNet/tree/main/docs/images/DeSeNet网络结构.png)

#### 2.1 SegMaskPSP(n_segcls=2, n=1, c_hide=256, shortcut=False, ch=()) 语义分割检测头

![](https://github.com/splwany/DeSeNet/tree/main/docs/images/SegMaskPSP结构.png)

接收三个输入，根据配置文件，分别来自第16层、19层、22层；

分别输入`m8`、`m16`、`m32`，`m16`、`m32`的输出分别经过2倍、4倍上采样恢复到跟`m8`的输出形状一样，然后在通道维上拼接，再输入到`out`层；

`out`层中，包含一个`RFB2`模块、一个`PyramidPooling`模块、一个`FFM`模块、一个通道数调整层、一个8倍上采样。

#### 2.2 RFB2(in_planes, out_planes, map_reduce=4, d[2, 3], has_global=False)

`map_reduce`表示`inter_planes`根据`in_planes`缩小的倍数；

内部有四个分支，分别叫做`branch0`、`branch1`、`branch2`、`branch3`；

如果`has_global = True`，则还有一个`branch4`，是一个全局平均池化层；

最后的结果会把4个或5个结果在通道维上拼接起来，再经过一个`CBL`层，返回结果。

#### 2.3 PyramidPooling(in_channels, k=[1, 2, 3, 6])

其中`k`为四个池化层的输出尺寸，分别为 1 * 1、2 * 2、3 * 3、6 * 6；

四个池化层分别为`pool1`、`pool2`、`pool3`、`pool4`；

还有四个卷积核大小为1 * 1 的`CBL`用来将通道数砍为原来的1/4，分别为`conv1`、`conv2`、`conv3`、`conv4`;

将输入`x`池化成四种尺度，再重新上采样回原来的尺寸，再在通道维上与输入`x`拼接，故通道数变为输入的2倍，返回结果。

#### 2.4 FFM(in_chan, out_chan, reduction=1, is_cat=True, k=1) 特征融合模块 Feature Fusion Model

其中`reduction`为瓶颈结构通道数相对于`out_chan`的缩小倍数；

内部有一个`convblk`层，是一个`CBL`；

内部有一个`channel_attention`层，包含一个平均池化、一个1 * 1瓶颈，瓶颈结构的两个卷积一个带普通激活层一个带`Sigmoid`激活层；

输入`x`先通过`convblk`得到一个`feature`，这个`feature`再通过`channel_attention`得到一个`attention`，将`feature`和`attention`矩阵点乘，得到加了注意力的`feature_attention`，最后将`feature_attention`和`feature`矩阵相加，返回结果。
