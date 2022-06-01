import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 10
# 生成数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # 输入数据
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))  # 真实值

torch_dataset = Data.TensorDataset(x, y)  # 数据处理
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE)  # 数据分批量集中

# 构建神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# 使用多种优化器
# 实例化四种模型
net_SGD = Net()
net_Momentum = Net()
net_RMSProp = Net()
net_Adam = Net()

nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]  # 四种实例化的模型

# 为每种模型定义优化器
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)
opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]  # 定义的优化器列表

# 训练模型
loss_fun = torch.nn.MSELoss()  # 定义损失函数
loss_his = [[], [], [], []]  # 记录四种模型+优化器的损失值
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, opt, l_his in zip(nets, optimizers, loss_his):
            output = net(batch_x)  # 输出
            loss = loss_fun(output, batch_y)  # 损失值
            opt.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            opt.step()  # 更新参数
            l_his.append(loss.data.numpy())  # 记录损失值
labels = ['SGD', 'Momentum', 'RMSProp', 'Adam']

# 可视化结果
for i, l_his in enumerate(loss_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('steps')
plt.ylabel('loss')
plt.ylim((0, 0.2))
plt.show()


