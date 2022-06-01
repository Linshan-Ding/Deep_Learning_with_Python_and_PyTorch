"""
快速梯度法实现无目标和目标攻击————AI新方向：对抗攻击
"""
"""
No objection attack
"""
import copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from torch.autograd.gradcheck import zero_gradients

import torchvision.transforms as T
from torchvision.models.inception import inception_v3

from PIL import Image

import matplotlib.pyplot as plt

import os
import numpy as np
from random import randint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择-GPU
classes = eval(open('classes.txt').read())  # 1000类集合
trans = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.unsqueeze(0))])  # 数据转换
reverse_trans = lambda x: np.asarray(T.ToPILImage()(x))  # 数据反转换

eps = 0.025
steps = 20
step_alpha = 0.01

model = inception_v3(pretrained=True, transform_input=True).to(device)  # 模型传入GPU
loss = nn.CrossEntropyLoss()
model.eval()

def load_image(img_path):
    """
    :param img_path: 图片存储路径
    :return: 转换为张量后的图片
    """
    img = trans(Image.open(img_path).convert('RGB'))
    return img

def get_class(img):
    """
    :param img: 模型输入的扰动后图片张量
    :return: 模型预测分类
    """
    with torch.no_grad():
        x = img.to(device)  # 传入GPU
        cls = model(x).data.max(1)[1].cpu().numpy()[0]  # 传回cpu改为numpy格式
        return classes[cls]

def draw_result(img, noise, adv_img):
    """
    :param img: 原始图片张量
    :param noise: 扰动因子张量
    :param adv_img: 扰动后图片张量
    :return: 保存图片
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    orig_class, attack_class = get_class(img), get_class(adv_img)
    ax[0].imshow(reverse_trans(img[0]))
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))
    ax[1].imshow(60*noise[0].detach().cpu().numpy().transpose(1, 2, 0))
    ax[1].set_title('Attacking noise')
    ax[2].imshow(reverse_trans(adv_img[0]))
    ax[2].set_title('Adversarial example: {}'.format(attack_class))
    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig('adv01.png', dpi=fig.dpi)

def non_targeted_attack(img):
    """
    无目标攻击
    :param img: 输入图片张量
    :return:
    """
    img = img.to(device)  # 传入GPU
    img.requires_grad = True  # 跟踪梯度
    label = torch.zeros(1, 1).to(device)

    x, y = img, label
    for step in range(steps):
        out = model(x)
        y = out.data.max(1)[1]  # 模型的实际预测结果
        local_loss = loss(out, y)  # 模型实际预测结果的损失函数
        local_loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)  # 输入梯度值
        step_adv = x.data + normed_grad
        adv = step_adv - img  # 更新样本误差
        adv = torch.clamp(adv, -eps, eps)  # 限制样本误差在一定范围
        # 更新图片像素值=x.data + normed_grad=next(x.data)
        # 加上梯度值使得损失函数值越来越大预测结果越来越不准确
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)  # 图片像素值范围限定
        x.data = result

    return result.cpu(), adv.cpu()

img = load_image('bird.JPEG')
adv_img, noise = non_targeted_attack(img)
draw_result(img, noise, adv_img)

"""
attack with a objection
"""
def targeted_attack(img, label):
    """
    目标攻击
    :param img: 输入图片张量
    :param label: 目标标签
    :return:
    """
    img = img.to(device)
    img.requires_grad = True
    label = torch.Tensor([label]).long().to(device)
    x, y = img, label
    for step in range(steps):
        out = model(x)
        local_loss = loss(out, y)  # 模型输出和目标标签损失值
        local_loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)  # 输入梯度
        step_adv = x.data - normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        # 更新图片像素值=x.data - normed_grad=next(x.data)
        # 减去梯度值使得损失函数值越来越小预测结果越来越像目标标签
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()

img = load_image('bird.JPEG')
adv_img, noise = targeted_attack(img, 600)
draw_result(img, noise, adv_img)

