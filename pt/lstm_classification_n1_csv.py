# -*- coding: utf-8 -*-
# 利用长短时记忆神经网络（LSTM）多分类，测试数据mnist的csv版本
# 输入数据具有三个维度，一般为[batch,time,width]，输出层有10个神经元
import torch
import torch.autograd as autograd
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
class LSTMModel(nn.Module):
    def __init__(self, input_dim, batch_size, layers_num, hidden_dim, class_size):
        super(LSTMModel, self).__init__()
        '''设置LSTM的超参数  start'''
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.class_size = class_size
        '''设置LSTM的超参数  end'''
        '''模型结构为LSTM层->全连接层'''
        # batch_first=True意味着输出维度为：[batch,time,data]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers_num, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, class_size)
    def init_hidden(self):
        # 初始化隐藏层
        return (autograd.Variable(torch.zeros(self.layers_num, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.layers_num, self.batch_size, self.hidden_dim)))

    def forward(self, x):
        # x维度：[batch,time,data]
        lstm_out, self.hidden = self.lstm(x, self.init_hidden())
        # 因为是many to one模型，所以取得最后一个时间戳的输出
        # 然后经过全连接层
        # 由于batch_first=True，所以最后一个时间戳的输出为lstm_out[:, -1, :]
        # 如果batch_first=False，最后一个时间戳的输出为lstm_out[-1, :, :]
        output = self.hidden2tag(lstm_out[:, -1, :].view(-1, self.hidden_dim))
        return output
'''--------定义模型 end--------'''
'''--------加载数据 start--------'''
# 加载训练数据
ds = CSVDataset()
# 构造训练迭代器
train_loader = DataLoader(dataset=ds,
                          batch_size=128,
                          shuffle=True,
                          drop_last=True)
# 加载测试数据，注意：dtype=np.float32，不然容易报错，类型不匹配
testData = np.loadtxt('../data/mnist_test.csv', delimiter=",", dtype=np.float32)
# 测试数据集的数据（将minist数据集看成一个时间序列，每一行是一个时间节点）
x_test = testData[0:128, 1:].reshape(-1, 28, 28)
x_test = x_test / 255
# 测试数据集的label
y_test = testData[0:128, 0]
'''--------加载数据 end--------'''

'''--------构造优化器及损失函数 start--------'''
# input_dim：28
# batch_size：128
# layers_num：1
# hidden_dim：64
# class_size：10
model = LSTMModel(28, 128, 1, 64, 10)
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
        output = model(batch_x.view(-1, 28, 28))
        # 3.计算损失
        loss = loss_fn(output, batch_y)
        # 4.反向传递，求变量w的梯度
        loss.backward()
        # 5.优化
        optimizer.step()
        if index % 10 == 0:
            # 每迭代10次，计算一次正确率
            y_pre = model(torch.Tensor(x_test))
            correct_prediction = np.equal(np.argmax(y_pre.detach().numpy(), 1), y_test)
            # 计算正确率
            # 注意correct_prediction.astype(np.float)，将booolean的tensor转为float类型
            accuracy = np.mean(correct_prediction.astype(np.float))
            print(accuracy)
'''--------训练模型 end--------'''