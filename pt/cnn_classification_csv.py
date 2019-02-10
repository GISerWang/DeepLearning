# -*- coding: utf-8 -*-
# 利用卷积神经网络多分类，测试数据mnist的csv版本
# 输入数据具有四个维度，一般为[batch,channels,height,width]，输出层有10个神经元
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
        # 将数据设置为四位矩阵：[batch,channels,height,width]
        self.x_data = torch.from_numpy(trainData[:, 1:]).view(-1, 1, 28, 28)
        self.y_data = torch.LongTensor(trainData[:, 0])
        # self.len是一共具有多少数据
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
class CNNModel(nn.Module):
    def __init__(self):
        # 卷积+标准化层+drop层+池化--->卷积+标准化层+drop层+池化--->全连接层+标准化层+drop层--->全连接层
        # conv->bn->drop->pool->conv->bn->drop->pool->fc->bn->drop->fc
        # 第一步，必须首先调用父类的构造方法
        super(CNNModel, self).__init__()
        # Conv2d：卷积层
        #   in_channels：输入图像的波段，为1
        #   out_channels：卷积核的个数：32
        #   kernel_size卷积核的大小
        #   stride：移动的步长，此参数可以是一个元组
        #   padding：填充的大小，填充的大小与卷积核的大小的关系为：2n+1=卷积核大小，n为填充大小
        self.conv1_fn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        # BatchNorm2d：批归一化层，此层必须放在非线性激励的前面
        #   num_features：为输入的特征数，针对图像num_features为波段数
        #   输入：[N,channels,height,width] - 输出：[N,channels,height,width]
        self.bn1_fn = nn.BatchNorm2d(32)
        # AvgPool2d：池化层
        #   kernel_size：池化窗口的大小
        #   stride：移动步长，设移动步长为k，那么featureMap的宽和高变为原来的1/k，
        #   padding：填充的大小，填充的大小与池化窗口的大小的关系为：2n+1=池化窗口大小，n为填充大小
        self.pool2_fn = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_fn = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2_fn = nn.BatchNorm2d(64)
        self.pool4_fn = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Linear：全连接层
        #   in_features ：输入特征数
        #   out_features ：输出的特征数
        #   输入的矩阵为：[32, 1000],输出的矩阵为：[32, 10]
        self.fc5_fn = nn.Linear(7 * 7 * 64, 1000)
        # BatchNorm1d：批归一化层，此层必须放在非线性激励的前面
        #   num_features：为输入的特征数，针对一维向量，隐藏层的个数
        #   输入：[N,num_features] - 输出：[N,num_features]
        self.bn3_fn = nn.BatchNorm1d(1000)
        self.out_layer_fn = nn.Linear(1000, 10)
        # Dropout：drop层，一般处于非线性激励的后面
        #   只定义了一个drop层，但是多次利用（drop没有参数，可以共享）
        self.drop_fn = nn.Dropout(0.2)
        # 自己初始化一些权重参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)
    def forward(self, x):
        # 卷积操作：输入的矩阵为：[32, 28, 28, 1],输出的矩阵为：[32, 28, 28, 32]
        conv1 = self.conv1_fn(x)
        # 批归一化操作，处于非线性激励的前面：输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 28, 28, 32]
        conv1 = self.bn1_fn(conv1)
        # 非线性激励：输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 28, 28, 32]
        conv1 = F.relu(conv1)
        # drop操作，处于非线性激励后面：输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 28, 28, 32]
        conv1 = self.drop_fn(conv1)
        # 池化操作：输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 14, 14, 32]
        pool2 = self.pool2_fn(conv1)
        # 卷积操作：输入的矩阵为：[32, 14, 14, 32],输出的矩阵为：[32, 14, 14, 64]
        conv3 = self.conv3_fn(pool2)
        # 批归一化操作，处于非线性激励的前面：输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 14, 14, 64]
        conv3 = self.bn2_fn(conv3)
        # 非线性激励：输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 14, 14, 64]
        conv3 = F.relu(conv3)
        # drop操作，处于非线性激励后面：输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 14, 14, 64]
        conv3 = self.drop_fn(conv3)
        # 池化操作：输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 7, 7, 64]
        pool4 = self.pool4_fn(conv3)
        # 将数据变为向量，为了全连接层
        pool4 = pool4.view(-1, 7 * 7 * 64)
        # 矩阵乘法：输入的矩阵为：[32, 7 * 7 * 64],输出的矩阵为：[32, 1000]
        fn5 = self.fc5_fn(pool4)
        # 批归一化操作，处于非线性激励的前面：[32, 1000],输出的矩阵为：[32, 1000]
        fn5 = self.bn3_fn(fn5)
        fn5 = F.relu(fn5)
        fn5 = self.drop_fn(fn5)
        # 矩阵乘法：输入的矩阵为：[32, 1000],输出的矩阵为：[32, 10]
        out = self.out_layer_fn(fn5)
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
x_test = testData[:, 1:].reshape(-1, 1, 28, 28)
# 测试数据集的label
y_test = testData[:, 0]
'''--------加载数据 end--------'''
'''--------构造优化器及损失函数 start--------'''
model = CNNModel()
# 将模型定义为训练模式，训练模式：drop生效，batch使用批均值
model.train()
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
            # 将模型设置为验证模式，验证模式：drop无效，batch使用滑动平均的定值
            model.eval()
            # 每迭代50次，计算一次正确率
            y_pre = model(torch.Tensor(x_test))
            correct_prediction = np.equal(np.argmax(y_pre.detach().numpy(), 1), y_test)
            # 计算正确率
            # 注意correct_prediction.astype(np.float)，将booolean的tensor转为float类型
            accuracy = np.mean(correct_prediction.astype(np.float))
            print(accuracy)
            # 将模型定义为训练模式，训练模式：drop生效，batch使用批均值
            model.train()
'''--------训练模型 end--------'''

