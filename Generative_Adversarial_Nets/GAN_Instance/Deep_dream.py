import copy
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

# 模型加载位置
import os
os.environ['TORCH_HOME'] = 'Models'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

def load_image(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.title("Image loaded successfully")
    return image

normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalise])

def deprocess(image):
    images = image * torch.tensor([0.229, 0.224, 0.225], device=device) + torch.tensor([0.485, 0.456, 0.406], device=device)
    return images

#下载预训练模型vgg19
vgg = models.vgg19(pretrained=True)
vgg = vgg.to(device)
print(vgg)
modulelist = list(vgg.features.modules())

def prod(image, layer, iterations, lr):
    """
    :param image: 图片
    :param layer: 传递到的网络层数
    :param iterations: 迭代次数
    :param lr: 学习率
    :return: 特征化后的输入图像
    """
    input = preprocess(image).unsqueeze(0)  # 三通道彩色图片
    input = input.to(device).requires_grad_(True)  # 输入图片像素值操作记录方面后续梯度求解
    vgg.zero_grad()  # 模型梯度清零
    for i in range(iterations):
        out = input
        for j in range(layer):
            out = modulelist[j + 1](out)
        # 以特征值的L2为损失值
        loss = out.norm()  # 计算输出的损失值
        loss.backward()  # 反向传播
        # 使梯度增大
        with torch.no_grad():  # 上下文环境切断自动求导运算后才可对相关参数进行处理
            input += lr * input.grad  # 根据输入梯度和学习率更新输入图像像素值
    input = input.squeeze()
    # 交互维度
    with torch.no_grad():  # 上下文环境切断自动求导运算后才可对相关参数进行处理
        input.transpose_(0, 1)
        input.transpose_(1, 2)
    # 使数据限制在[0,1]之间
    input = np.clip(deprocess(input).detach().cpu().numpy(), 0, 1)
    im = Image.fromarray(np.uint8(input * 255))
    return im

def deep_dream_vgg(image, layer, iterations, lr, octave_scale=2, num_octaves=20):
    """
    递归调用函数
    :param octave_scale: 每次图片缩小的倍数
    :param num_octaves: 图片缩小次数
    :return: 输出变化后的输入图片从而展示模型学习的特征
    """
    if num_octaves > 0:
        image1 = image.filter(ImageFilter.GaussianBlur(2))
        if (image1.size[0] / octave_scale < 1 or image1.size[1] / octave_scale < 1):
            size = image1.size
        else:
            size = (int(image1.size[0] / octave_scale), int(image1.size[1] / octave_scale))

        image1 = image1.resize(size, Image.ANTIALIAS)  # 缩小图片
        image1 = deep_dream_vgg(image1, layer, iterations, lr, octave_scale, num_octaves - 1)   # 函数递归调用
        size = (image.size[0], image.size[1])
        image1 = image1.resize(size, Image.ANTIALIAS)  # 放大图片
        image = ImageChops.blend(image, image1, 0.6)  # 图像组合
    img_result = prod(image, layer, iterations, lr)  # 图片像素值更新：Deep Dream
    img_result = img_result.resize(image.size)  # 图片尺寸调整
    plt.imshow(img_result)
    return img_result

night_sky = load_image('data/starry_night.jpg')
# night_sky.show()

night_sky_32 = deep_dream_vgg(night_sky, 32, 6, 0.2)
night_sky_32.show()

night_sky_8 = deep_dream_vgg(night_sky, 8, 6, 0.2)
night_sky_8.show()

night_sky_4 = deep_dream_vgg(night_sky, 4, 6, 0.2)
night_sky_4.show()