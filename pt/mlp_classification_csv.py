# -*- coding: utf-8 -*-
# 利用多层感知机多分类，测试数据为在线的minist图像
# 输入层具有28*28个神经元，输出层有10个神经元
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
'''--------定义数据加载类 start--------'''
class CSVDataset(Dataset):
    # 1.继承Dataset父类
    # 2.从构造函数中加载数据（文件名可以通过构造函数传进来）
    # 3.重写__getitem__方法，此方法默认有一个index参数，返回相应的数据及其类别
    # 4.重写__len__方法，返回数据的个数，即len
    def __init__(self):
        trainData = np.loadtxt('../data/mnist_train.csv', delimiter=",", dtype=np.float32)
        # 标准化数据，不然不容易收敛
        self.x_data = torch.from_numpy(trainData[:, 1:]/255)
        self.y_data = torch.LongTensor(trainData[:, 0])
        self.len = trainData.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
'''--------定义数据加载类 end--------'''

'''--------定义模型 start--------'''
# 1.继承nn.Module父类
# 2.重写构造函数，定义模型结构
# 3.重写forward函数，定义模型的前馈过程
class MLPModel(nn.Module):
    def __init__(self):
        '''模型结构为784->32->64->32->10'''
        # 必须要调用父类的构造方法
        super(MLPModel, self).__init__()
        # 定义第一个全连接层，784个输入，32个输出
        self.layer1 = nn.Linear(784, 32)
        # 定义第二个全连接层，32个输入，64个输出
        self.layer2 = nn.Linear(32, 64)
        # 定义第三个全连接层，64个输入，32个输出
        self.layer3 = nn.Linear(64, 32)
        # 定义最后一个全连接层，32个输入，10个输出
        self.out_layer = nn.Linear(32, 10)
    def forward(self, x):
        # 定义模型的前馈，使用relu作为激活函数
        layer1_out = F.relu(self.layer1(x))
        # 定义模型的前馈，使用relu作为激活函数
        layer2_out = F.relu(self.layer2(layer1_out))
        # 定义模型的前馈，使用relu作为激活函数
        layer3_out = F.relu(self.layer3(layer2_out))
        # 定义模型的前馈，使用relu作为激活函数
        out = self.out_layer(layer3_out)
        return out
'''--------定义模型 end--------'''
'''--------加载数据 start--------'''
# 加载训练数据
ds = CSVDataset()
# 构造训练迭代器
train_loader = DataLoader(dataset=ds,
                          batch_size=32,
                          shuffle=True)
# 加载测试数据，注意：dtype=np.float32，不然容易报错，类型不匹配
testData = np.loadtxt('../data/mnist_test.csv',delimiter=",", dtype=np.float32)
# 测试数据集的数据
x_test = testData[:, 1:]/255
# 测试数据集的label
y_test = testData[:, 0]
'''--------加载数据 end--------'''
'''--------构造优化器及损失函数 start--------'''
model = MLPModel()
# 构造Adam优化器，用于优化模型的参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 构造交叉熵损失函数
# CrossEntropyLoss自带softmax方法
# 因此神经网络的最后一层，不需要softmax，一般不需要激活函数
loss_fn = torch.nn.CrossEntropyLoss()
'''--------构造优化器及损失函数 end--------'''

'''--------训练模型 start--------'''
while True:
    for index, (batch_x, batch_y) in enumerate(train_loader):
        # 1.优化器将梯度归零（必须设置）
        optimizer.zero_grad()
        # 2.计算模型的输出
        output = model(batch_x)
        # 3.计算损失
        loss = loss_fn(output, batch_y)
        # 4.反向传递，求变量w的梯度
        loss.backward()
        # 5.优化
        optimizer.step()
        if index % 50 == 0:
            # 每迭代50次，计算一次正确率
            y_pre = model(torch.Tensor(x_test))
            correct_prediction = np.equal(np.argmax(y_pre.detach().numpy(), 1), y_test)
            # 计算正确率
            # 注意correct_prediction.astype(np.float)，将booolean的tensor转为float类型
            accuracy = np.mean(correct_prediction.astype(np.float))
            print(accuracy)
'''--------训练模型 end--------'''

