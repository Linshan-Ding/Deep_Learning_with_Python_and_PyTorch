'''
导入库文件---可视化训练
'''
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
from visdom import Visdom
import numpy as np


'''
构建简单的模型:简单线性层+Relu函数的多层感知机
'''
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


batch_size = 128
learning_rate = 0.01
epochs = 10

# 注意数据集路径
train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    r'D:\Users\Administrator\Desktop\PythonDLbasedonPytorch\data',
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))])),
                                            batch_size=batch_size,
                                            shuffle=True)
# 注意数据集路径
test_loader = torch.utils.data.DataLoader(datasets.MNIST(
    r'D:\Users\Administrator\Desktop\PythonDLbasedonPytorch\data',
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))])),
                                            batch_size=batch_size,
                                            shuffle=True)

# 注意此处初始化visdom类
viz = Visdom()
# 绘制起点
viz.line([0.], [0.], win="train loss", opts=dict(title='train_loss'))
device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(test_loader.dataset)
    # 绘制epoch以及对应的测试集损失loss
    viz.line([test_loss], [epoch], win="train loss", update='append')
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
