import numpy as np
import torch
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 实例化存放路径
writer = SummaryWriter(log_dir='logs', comment='Linear')
# 定义超参数
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoches = 5
lr = 0.01
momentum = 0.5
# 下载并对数据进行预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # 数据处理函数集合
train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=False)  # 训练数据下载
test_dataset = mnist.MNIST('./data', train=False, transform=transform, download=False)  # 测试数据下载
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)  # 训练数据迭代对象
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)  # 测试数据迭代对象

# 可视化源数据
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
fig=plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
    plt.title("Ground Truth:{}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])

# 构建模型
# 构建网络
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
# 实例化网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检测是否有可用的GPU，有则使用，否则使用CPU
model = Net(28*28, 300, 100, 10)
model.to(device)  # 把模型发送到GPU/CPU上
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 定义优化器

# 训练模型
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
# 每个周期开始时刻以上个周期结束时刻优化后的参数为优化起点
for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1
    for img, label in train_loader:
        img = img.to(device)  # 把训练数据发送到GPU/CPU上
        label = label.to(device)  # 把训练数据发送到GPU/CPU上
        img = img.view(img.size(0), -1)  # 矩阵展平
        # 前向传播
        out = model(img)  # 输出层10列的输出值（激活值）最大激活值对应索引做为最终输出: out = model.forward(img)
        loss = criterion(out, label)  # out(最大激活值对应索引)总的损失函数值
        # 反向传播
        optimizer.zero_grad()  # 参数梯度清零
        loss.backward()  # 求梯度
        optimizer.step()  # 优化器更新权重
        # 记录误差 （所有批量数据累加的损失函数值）
        train_loss += loss.item()
        # 计算分类的准确率 （所有批量数据累加准确率)
        _, pred = out.max(1)  # 找出每列的最大值(激活值)对应索引(数字0-9)做为最终输出
        num_correct = (pred == label).sum().item()  # 输出值和实际值符合数
        acc = num_correct/img.shape[0]  # 占比
        train_acc += acc

    losses.append(train_loss/len(train_loader))  # 记录本周期训练总的损失率
    acces.append(train_acc/len(train_loader))  # 记录本周期训练总的准确率

    # 保存loss的数据与epoch数值
    writer.add_scalar('训练损失值', train_loss, epoch)

    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    # 模型测试阶段
    model.eval()  # 改为预测模型
    for img, label in test_loader:
        img = img.to(device)  # 把测试数据发送到GPU/CPU上
        label = label.to(device)  # 把测试数据发送到GPU/CPU上
        img = img.view(img.size(0), -1)
        # 前向传播
        out = model(img)  # 输出
        loss = criterion(out, label)  # 计算损失函数值
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()  # 预测值和实际值符合数
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss/len(test_loader))
    eval_acces.append(eval_acc/len(test_loader))
    print('epoch:{}, Train Loss:{:.4f}, Train Acc:{:.4f}, Test Loss:{:.4f}, Test Acc:{:.4f}'.format(epoch, train_loss/len(train_loader),
                                                                                                    train_acc/len(train_loader),
                                                                                                    eval_loss/len(test_loader),
                                                                                                    eval_acc/len(test_loader)))
print('训练误差率：{}'.format(losses))
print('训练准确率：{}'.format(acces))
print('测试误差：{}'.format(eval_losses))
print('测试准确：{}'.format(eval_acces))

# 可视化训练结果
plt.title('trainloss')
plt.plot(np.arange(len(losses)), losses)
plt.legend(['Train Loss'], loc='upper right')
plt.show()
