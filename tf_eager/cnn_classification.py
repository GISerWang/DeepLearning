# -*-coding: utf-8 -*-
# 利用卷积神经网络多分类，测试数据mnist的csv版本，抽样版本(eager模式，即tensorflow动态图机制)
# 使用tensorflow高级API进行多分类任务
# 输入数据具有四个维度，一般为[batch,height,width,channels]，输出层有10个神经元
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 启动tensorflow的动态图机制
tfe.enable_eager_execution()
'''--------加载数据start--------'''
# 加载训练数据及测试数据
trainData = np.loadtxt('../data/mnist_train.csv',delimiter=",")
testData = np.loadtxt('../data/mnist_test.csv',delimiter=",")
# 提取训练数据的x 及 y
# 将数据转换为[batch,height,width,channels]
x_train = trainData[:, 1:].reshape(-1, 28, 28, 1)
y_train = trainData[:, 0]
# 将y_train转换为one-hot形式
y_train_one_hot = tf.one_hot(y_train.astype(np.int), 10).numpy()
# 提取测试数据的x,y
# 将数据转换为[batch,height,width,channels]
x_test = testData[:, 1:].reshape(-1, 28, 28, 1)
y_test = testData[:, 0]
# 将训练数据封装到dataSet中
# shuffle意味着打乱训练数据集中的顺序
# repeat意味着训练数据无限重复
# batch意味着每次在dataSet中提取512条数据
ds = tf.data.Dataset.from_tensor_slices({
    "x": x_train,
    'y': y_train_one_hot
}).shuffle(buffer_size=123).repeat().batch(64)
'''--------加载数据end--------'''
'''--------模型start--------'''
# 卷积+归一化+drop+池化+卷积+归一化+drop+池化+全连接层+归一化+drop+全连接层即：
# conv->bn->drop->pool->conv->bn->drop->pool->fc->bn->drop->fc
class CNNModel(object):
    def __init__(self):
        # 定义卷积层
        # strides = [stride, stride]:stride是卷积核的移动步长
        # padding可以是SAME也可以是VALID
        # SAME表示填充，填充的大小与卷积核的大小的关系为：2n+1=卷积核大小，n为填充大小
        # kernel_size定义卷积核的大小，卷积核的长为5，宽为5，
        # filters卷积核的个数为32个
        # 输入图像的波段被省略，即channels被省略
        self.conv1_fn = tf.layers.Conv2D(filters=32, kernel_size=5, strides=[1, 1], padding='SAME')
        # 进行归一化操作，归一化操作定义在非线性激励之前
        # axis是归一化的轴，当数据格式为[batch,height,width,channels]，axis=3
        self.bn1_fn = tf.layers.BatchNormalization(axis=3)
        # 创建第二层：池化层
        # pool_size为池化核的宽和高
        # strides = [stride, stride]:stride是卷积核的移动步长
        # 设移动步长为k，那么卷积核的宽和高变为原来的1/k，
        self.pool2_fn = tf.layers.AveragePooling2D(pool_size=3, strides=[2, 2], padding='SAME')
        # 定义卷积层
        # strides = [stride, stride]:stride是卷积核的移动步长
        # padding可以是SAME也可以是VALID
        # SAME表示填充，填充的大小与卷积核的大小的关系为：2n+1=卷积核大小，n为填充大小
        # kernel_size定义卷积核的大小，卷积核的长为5，宽为5，
        # filters卷积核的个数为64个
        # 输入图像的波段被省略，即channels被省略
        self.conv3_fn = tf.layers.Conv2D(filters=64, kernel_size=5, strides=[1, 1], padding='SAME')
        # 进行归一化操作，归一化操作定义在非线性激励之前
        # axis是归一化的轴，当数据格式为[batch,height,width,channels]，axis=3
        self.bn2_fn = tf.layers.BatchNormalization(axis=3)
        # 创建池化层
        # pool_size为池化核的宽和高
        # strides = [stride, stride]:stride是卷积核的移动步长
        # 设移动步长为k，那么卷积核的宽和高变为原来的1/k，
        self.pool4_fn = tf.layers.AveragePooling2D(pool_size=3, strides=[2, 2], padding='SAME')
        # 定义全连接层
        # 权重的形状为[in,out]
        # 其中in等于上一层特征图，经拉直后的大小即：[32, 7, 7, 64]->[32, 7*7*64]
        # out为下一层全连接层的个数
        # bias的大小与out相等
        self.fc5_fn = tf.layers.Dense(units=1000)
        # 进行归一化操作，归一化操作定义在非线性激励之前
        # axis是归一化的轴，当数据格式为[batch,height,width,channels]，axis=3
        self.bn3_fn = tf.layers.BatchNormalization()
        self.out_layer_fn = tf.layers.Dense(units=10)
        # 定义dropout层，将drop层进行共享
        self.drop_fn = tf.layers.Dropout(rate=0.2)

    def forward(self, inputs, training):
        # 输入的矩阵为：[32, 28, 28, 1],输出的矩阵为：[32, 28, 28, 32]
        conv1 = self.conv1_fn(inputs)
        # 输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 28, 28, 32]
        conv1 = self.bn1_fn(conv1, training)
        # 输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 28, 28, 32]
        conv1 = tf.nn.relu(conv1)
        # 输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 28, 28, 32]
        conv1 = self.drop_fn(conv1, training)
        # 输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 14, 14, 32]
        pool2 = self.pool2_fn(conv1)
        # 输入的矩阵为：[32, 14, 14, 32],输出的矩阵为：[32, 14, 14, 64]
        conv3 = self.conv3_fn(pool2)
        # 输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 14, 14, 64]
        conv3 = self.bn2_fn(conv3, training)
        # 输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 14, 14, 64]
        conv3 = tf.nn.relu(conv3)
        # 输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 14, 14, 64]
        conv3 = self.drop_fn(conv3, training)
        # 输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 7, 7, 64]
        pool4 = self.pool4_fn(conv3)
        # 输入的矩阵为：[32, 7, 7, 64],输出的矩阵为：[32, 7*7*64]
        pool4 = tf.reshape(pool4, [-1, 7*7*64])
        # 输入的矩阵为：[32, 7*7*64],输出的矩阵为：[32, 1000]
        fn5 = self.fc5_fn(pool4)
        # 输入的矩阵为：[32, 1000],输出的矩阵为：[32, 1000]
        fn5 = self.bn3_fn(fn5, training)
        # 输入的矩阵为：[32, 1000],输出的矩阵为：[32, 1000]
        fn5 = tf.nn.relu(fn5)
        # 输入的矩阵为：[32, 1000],输出的矩阵为：[32, 1000]
        fn5 = self.drop_fn(fn5, training)
        # 输入的矩阵为：[32, 1000],输出的矩阵为：[32, 10]
        out = self.out_layer_fn(fn5)
        return out
'''--------模型end--------'''

'''--------定义loss函数 start--------'''
# 必须在loss_fn函数里面进行前项传播，然后经implicit_gradients包装后，才能进行梯度计算
def loss_fn(forward, inputs, ys,training):
    # tf.nn.softmax_cross_entropy_with_logits_v2等价于tf.nn.softmax+cross_entropy
    #       1.最后一层不需要使用激活函数softmax，直接使用softmax_cross_entropy_with_logits_v2即可
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=forward(inputs,training)))
'''--------定义loss函数 end--------'''

'''--------定义优化器与梯度函数 start--------'''
# ys与out_layer都是one-hot类型的向量
optimizer = tf.train.AdamOptimizer(0.01)
grad_fn = tfe.implicit_gradients(loss_fn)
'''--------定义优化器与梯度函数 end--------'''

'''--------训练start--------'''
model = CNNModel()
for index, batchData in enumerate(tfe.Iterator(ds)):
    # 计算梯度，grad是当前的梯度数组
    grad = grad_fn(model.forward, tf.cast(batchData['x'], dtype=tf.float32), batchData['y'], True)
    # 优化权重，反向传播
    optimizer.apply_gradients(grad)
    if index % 50 == 0:
        # 测试数据
        y_pre = model.forward(tf.cast(x_test, dtype=tf.float32), False)
        # 测试正确率
        correct_prediction = np.equal(np.argmax(y_pre.numpy(), 1), y_test)
        accuracy = np.mean(correct_prediction.astype(np.float))
        print(accuracy)
'''--------训练end--------'''
