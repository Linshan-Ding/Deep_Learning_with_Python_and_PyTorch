# 可视化数据集

from visdom import Visdom
import numpy as np
import torch
from torchvision import datasets, transforms

# 注意数据集路径
train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    r'./data',
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor()])),batch_size=128,shuffle=True)
sample=next(iter(train_loader)) # 通过迭代器获取样本
# sample[0]为样本数据 sample[1]为类别  nrow=16表示每行显示16张图像
viz = Visdom(env='my_visual') # 注意此时我已经换了新环境
viz.images(sample[0],nrow=16,win='mnist',opts=dict(title='mnist'))

# 'mnist'
