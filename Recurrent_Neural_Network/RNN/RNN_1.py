# 前向传播与随时间反向传播
import numpy as np

X = [1,2]  # 两个时间点的输入
state = [0.0, 0.0]  # 初始化状态值
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 初始化（输入-隐含层）+（状态-隐含层）权重
b_cell = np.asarray([0.1, -0.1])  # 初始化偏移量
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

for i in range(len(X)):
    state=np.append(state,X[i])
    before_activation = np.dot(state, w_cell_state) + b_cell
    state = np.tanh(before_activation)  # 激活函数--新状态
    final_output = np.dot(state, w_output) + b_output  # 该时间点输出
    print("状态值_%i: "%i, state)
    print("输出值_%i: "%i, final_output)
