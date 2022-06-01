from IPython.core.interactiveshell import InteractiveShell
# Data science tools
import numpy as np
# Image manipulations
from PIL import Image
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
#  %matplotlib inline
plt.rcParams['font.size'] = 14
# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'

def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
# Example image
# x = Image.open('./image/cat/cat.jpg')
# print(np.array(x).shape)
# imshow(x)

import sys
from PIL import Image
from torchvision import transforms as trans
im = Image.open('./image/cat/wa.jpg')
# imshow(im)

# 随机比例缩放主要使用的是 torchvision.transforms.Resize()
# 比例缩放
print('原图片大小: {}'.format(im.size))
new_im = trans.Resize((100, 200))(im)
print('缩放后大小: {}'.format(new_im.size))
# imshow(new_im)

# 随机位置截取能够提取出图片中局部的信息，使得网络接受的输入具有多尺度的特征，所以能够有较好的效果。
# 在 torchvision 中主要有下面两种方式，一个是 torchvision.transforms.RandomCrop()，
# 传入的参数就是截取出的图片的长和宽，对图片在随机位置进行截取；第二个是 torchvision.transforms.CenterCrop()，
# 同样传入介曲初的图片的大小作为参数，会在图片的中心进行截取
# 随机裁剪出 100 x 100 的区域
# random_im1 = trans.RandomCrop(200)(im)
# imshow(random_im1)

# 随机竖直翻转
# v_flip = trans.RandomVerticalFlip()(im)
# imshow(v_flip)

# 旋转45°
# rot_im = trans.RandomRotation(45)(im)
# imshow(rot_im)

# 除了形状变化外，颜色变化又是另外一种增强方式，其中可以设置亮度变化，对比度变化和颜色变化等，
# 在 torchvision 中主要使用 torchvision.transforms.ColorJitter() 来实现的，第一个参数就是亮度的比例，
# 第二个是对比度，第三个是饱和度，第四个是颜色
# 对比度
# contrast_im = trans.ColorJitter(contrast=1)(im)  # 随机从 0 ~ 2 之间对比度变化，1 表示原图
# imshow(contrast_im)
# 颜色
# color_im = trans.ColorJitter(hue=0.5)(im)  # 随机从 -0.5 ~ 0.5 之间对颜色变化
# imshow(color_im)

# 联合改变
im_aug = trans.Compose([trans.Resize(200), trans.RandomHorizontalFlip(),
                        trans.RandomCrop(96), trans.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)])
import matplotlib.pyplot as plt
nrows = 3
ncols = 3
figsize = (8, 8)
_, figs = plt.subplots(nrows, ncols, figsize=figsize)
plt.axis('off')
for i in range(nrows):
    for j in range(ncols):
        figs[i][j].imshow(im_aug(im))
plt.show()
