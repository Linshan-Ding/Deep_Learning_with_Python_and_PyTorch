# 多条曲线实时绘制

from visdom import Visdom
import numpy as np
import time

# 将窗口类实例化
viz = Visdom(env='my_loss')
# 创建窗口并初始化
viz.line([[0., 0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))

for global_steps in range(10):
    # 随机获取loss和acc
    loss = 0.1 * np.random.randn() + 1
    acc = 0.1 * np.random.randn() + 0.5
    # 更新窗口图像
    viz.line([[loss, acc]], [global_steps], win='train', update='append')
    # 延时0.5s
    time.sleep(0.5)