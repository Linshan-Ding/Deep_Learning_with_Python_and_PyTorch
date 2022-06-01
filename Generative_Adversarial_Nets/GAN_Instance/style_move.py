from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

# 指定输出图像大小
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
imsize_w = 600

# 对图像进行预处理
loader = transforms.Compose([transforms.Resize((imsize, imsize_w)), transforms.ToTensor()])

def image_loader(image_name):
    """
    图片转换为张量
    :param image_name: 导入的图片地址和名字
    :return: 张量表示的图片
    """
    image = Image.open(image_name)
    # 增加一个维度，其值为1，这是为了满足神经网络对输入图像的形状要求
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./data/starry_night.jpg")
content_img = image_loader("./data/dancing.jpg")

print("style size:", style_img.size())
print("content size:", content_img.size())
assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"

unloader = transforms.ToPILImage()  # 张量转换为图片

plt.ion()

def imshow(tensor, title=None):
    """
    张量转换为图片
    :param tensor: 张量
    :param title: 图片题目
    :return: 图片显示
    """
    image = tensor.cpu().clone()  # 为避免因image修改影响tensor的值，这里采用clone
    image = image.squeeze(0)      # 去掉批量这个维度
    image = unloader(image)       # 张量转换为图片
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    # image.show()  # 展示图片

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):
    # 内容损失函数
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 必须要用detach来分离出target，这时候target不再是一个Variable，这是为了动态计算梯度，否则forward会出错，不能向前传播.
        self.target = target.detach()  # 内容图片做为标签值模仿这个内容

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)  # 内容损失值
        return input

def gram_matrix(input):
    """
    格拉姆矩阵生成
    :param input: 输入图片张量
    :return: 标准化格拉姆矩阵
    """
    a, b, c, d = input.size()  # a表示批量（batch size）的大小，这里batch size=1
    # b是特征图的数量，(c,d)是特征图的维度(N=c*d)
    features = input.view(a * b, c * d)  # 对应图12-5中的x矩阵
    G = torch.mm(features, features.t())  # 计算内积
    # 对格拉姆矩阵标准化，通过对其处以特征图像素总数.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    # 风格损失函数
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()  # 风格图片做为标签值模仿这个风格

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)  # 风格损失值
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)  # 均值
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)  # 标准差

# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    """
    输入图片像素值标准化
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # view the mean and std to make them [C x 1 x 1] so that
        # they can directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.mean = mean.clone().detach().view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# 为计算内容损失和风格损失，指定使用的卷积层
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img,
                               content_layers=content_layers_default, style_layers=style_layers_default):
    """
    :param cnn: 导入的VGG19模型
    :param normalization_mean: 均值
    :param normalization_std: 方差
    :param style_img: 风格图片
    :param content_img: 内容图片
    :param content_layers: 内容层
    :param style_layers: 风格层
    :return: 构建的风格迁移深度学习模型+风格损失层对象列表+内柔损失层对象列表
    """
    cnn = copy.deepcopy(cnn)
    # 标准化模型
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # 初始化损失值
    content_losses = []
    style_losses = []
    # 使用sequential方法构建模型
    model = nn.Sequential(normalization)  # 添加标准化处理层
    i = 0  # 每次迭代增加1
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 加内容损失
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 累加风格损失
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 我们需要对在内容损失和风格损失之后的层进行修剪
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# copy the content_img
input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')

def get_input_optimizer(input_img):
    """
    :param input_img: 输入图片张量
    :return: 定义好的优化器对象
    """
    # 这里需要对输入图像进行梯度计算,故需要设置为requires_grad_()，优化方法采用LBFGS
    optimizer = optim.LBFGS([input_img.requires_grad_()])  # 输入像素值梯度可计算
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img,
                       input_img, num_steps=600, style_weight=1000000, content_weight=1):
    """
    风格转换主函数
    """
    print('Building the style transfer model..')  # 构建风格转换模型
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean,
                                                                     normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)  # 输入图片的值控制在【0， 1】
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            # 依据内容损失和风格损失权重更新对应损失值
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)
plt.figure()
imshow(output, title='Output Image')
# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()