# -*- coding: utf-8 -*-
# 利用多层感知机拟合回归曲线，曲线方程为：y=2*x^2 + 4*x - 0.5
# 输入层仅有一个神经元，输出层仅有一个神经元，即一对一（x->y），还可以运行一对多，多对多，多对一
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
''''''
trainData = np.loadtxt('../data/mlp_regression_train.csv',delimiter=",", dtype=np.float32)
testData = np.loadtxt('../data/mlp_regression_test.csv',delimiter=",", dtype=np.float32)
# 提取训练数据的x 及 y
x_train = torch.from_numpy(trainData[:, 0].reshape(-1, 1))
y_train = torch.from_numpy(trainData[:, 1].reshape(-1, 1))
x_test = torch.from_numpy(testData[:, 0].reshape(-1, 1))
y_test = torch.from_numpy(testData[:, 1].reshape(-1, 1))
# 将内存中的ndarray封装成Dataset，便于使用迭代器
ds = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=ds, batch_size=32, shuffle=True)
'''--------定义模型start--------'''
# 继承指定类：nn.Module
class MLPModel(nn.Module):
    def __init__(self):
        '''模型结构为1->4->5->3->1'''
        super(MLPModel, self).__init__()
        # 定义相关的全连接层参数
        self.layer1 = nn.Linear(1, 4)
        self.layer2 = nn.Linear(4, 5)
        self.layer3 = nn.Linear(5, 3)
        self.out_layer = nn.Linear(3, 1)
        # 为了收敛的速度，自己初始化权重
        # 因此本示例仅有全连接层，因此仅对全连接层初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)
    def forward(self, x):
        # 神经网络的前项传播
        # 由于使用数据存在小于0点数字，因此采用leaky_relu作为激活函数
        layer1_out = F.leaky_relu(self.layer1(x), negative_slope=0.2)
        layer2_out = F.leaky_relu(self.layer2(layer1_out), negative_slope=0.2)
        layer3_out = F.leaky_relu(self.layer3(layer2_out), negative_slope=0.2)
        out = self.out_layer(layer3_out)
        return out
'''--------定义模型end--------'''

'''--------绘制训练数据start--------'''
# 绘制训练数据
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(trainData[:, 0], trainData[:, 1])
# 设置：弹出窗口，但是并不阻塞进程
plt.ion()
plt.show()
'''--------绘制训练数据end--------'''
'''--------定义优化器与损失函数 start--------'''
# 定义模型以及优化器和损失函数
model = MLPModel()
# Stochastic Gradient Descent随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
# 使用均方差公式作为损失函数
loss_fn = torch.nn.MSELoss()

'''--------迭代 start--------'''
while True:
    for index, (batch_x, batch_y) in enumerate(train_loader):
        # 第一步，首先将梯度归零
        optimizer.zero_grad()
        # 第二步，使用模型计算输出
        output = model(batch_x)
        # 第三步，使用损失函数计算loss
        loss = loss_fn(output, batch_y)
        # 第四步，使用bp计算梯度
        loss.backward()
        # 第五步，优化权重
        optimizer.step()
        if index % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception as e:
                print(e)
            prediction = model(x_test)
            lines = ax.plot(testData[:, 0], prediction.detach().numpy(), 'r-', lw=1)
            # 测试数据的loss
            print(loss_fn(prediction, y_test))
            plt.pause(0.1)
'''--------迭代 end--------'''

