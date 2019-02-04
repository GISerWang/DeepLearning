# -*- coding: utf-8 -*-
# 利用多层感知机多分类，测试数据为在线的minist图像
# 输入层具有28*28个神经元，输出层有10个神经元
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import dataloader
import torch.nn.functional as F
import numpy as np
'''--------定义模型start--------'''
class MLPModel(nn.Module):
    def __init__(self):
        '''模型结构为784->32->64->32->10'''
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(784, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 32)
        self.out_layer = nn.Linear(32, 10)
    def forward(self, x):
        # 定义模型的前馈，使用relu作为激活函数
        layer1_out = F.relu(self.layer1(x))
        # 定义模型的前馈，使用relu作为激活函数
        layer2_out = F.relu(self.layer2(layer1_out))
        # 定义模型的前馈，使用relu作为激活函数
        layer3_out = F.relu(self.layer3(layer2_out))
        # 定义模型的前馈，没有使用激活函数
        out = self.out_layer(layer3_out)
        return out
'''--------定义模型end--------'''

'''--------加载数据start--------'''
train_data = torchvision.datasets.MNIST(
    root='../data/mnist_pytorch/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)
test_data = torchvision.datasets.MNIST(
    '../data/mnist_pytorch/', train=False, transform=torchvision.transforms.ToTensor()
)
train_loader = dataloader.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_x = test_data.test_data.view(-1, 784)
'''--------加载数据end--------'''

'''--------构造优化器及损失函数 start--------'''
model = MLPModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
'''--------构造优化器及损失函数 end--------'''
for epoch in range(100):
    for index, (batch_x, batch_y) in enumerate(train_loader):
        # 1.优化器将梯度归零（必须设置）
        optimizer.zero_grad()
        batch_x = batch_x.view(-1, 784)
        data, target = batch_x, batch_y
        # 2.计算模型的输出
        output = model(data)
        # 3.计算损失
        loss = loss_fn(output, target)
        # 4.反向传递，求变量w的梯度
        loss.backward()
        # 5.优化
        optimizer.step()
        if index % 50 == 0:
            # 每迭代50次，计算一次正确率
            y_pre = model(test_x.type(torch.FloatTensor))
            # 计算正确率
            # 注意correct_prediction.astype(np.float)，将booolean的tensor转为float类型
            correct_prediction = np.equal(np.argmax(y_pre.detach().numpy(), 1), test_data.test_labels.numpy())
            accuracy = np.mean(correct_prediction.astype(np.float))
            print(accuracy)


