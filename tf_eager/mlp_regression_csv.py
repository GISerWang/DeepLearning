# -*- coding: utf-8 -*-
# eager模式，即tensorflow动态图机制
# 利用多层感知机拟合回归曲线，曲线方程为：y=2*x^2 + 4*x - 0.5
# 输入层仅有一个神经元，输出层仅有一个神经元，即一对一（x->y），还可以运行一对多，多对多，多对一
import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
# 启动tensorflow的动态图机制,仅能启动一次
tfe.enable_eager_execution()
'''--------加载数据start--------'''
# 加载训练数据及测试数据
trainData = np.loadtxt('../data/mlp_regression_train.csv',delimiter=",")
testData = np.loadtxt('../data/mlp_regression_test.csv',delimiter=",")
# 提取训练数据的x 及 y
x_train = trainData[:, 0].reshape(-1, 1)
y_train = trainData[:, 1].reshape(-1, 1)
# 提取测试数据的x,y
x_test = testData[:, 0].reshape(-1, 1)
y_test = testData[:, 1].reshape(-1, 1)
# 将训练数据封装到dataSet中
# shuffle意味着打乱训练数据集中的顺序
# repeat意味着训练数据无限重复
# batch意味着每次在dataSet中提取256条数据
ds = tf.data.Dataset.from_tensor_slices({
    "x": x_train,
    'y': y_train
}).shuffle(buffer_size=123).repeat().batch(256)
'''--------加载数据end--------'''

'''--------绘制训练数据start--------'''
# 绘制训练数据
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_train, y_train)
# 设置：弹出窗口，但是并不阻塞进程
plt.ion()
plt.show()
'''--------绘制训练数据end--------'''

'''--------模型start--------'''
class MLPModel(object):
    def __init__(self):
        # 定义第一层的weight：weight的维度为[in_size,out_size]
        self.w1 = tfe.Variable(tf.random_normal([1, 4]), name='w1')
        # 定义第一层的bias:bias的维度固定为[1,out_size]
        self.b1 = tfe.Variable(tf.random_normal([1, 4]), name='b1')
        # 定义第二层的weight：weight的维度为[in_size,out_size]
        self.w2 = tfe.Variable(tf.random_normal([4, 5]), name='w2')
        # 定义第二层的bias:bias的维度固定为[1,out_size]
        self.b2 = tfe.Variable(tf.random_normal([1, 5]), name='b2')
        # 定义第三层的weight：weight的维度为[in_size,out_size]
        self.w3 = tfe.Variable(tf.random_normal([5, 3]), name='w3')
        # 定义第三层的bias:bias的维度固定为[1,out_size]
        self.b3 = tfe.Variable(tf.random_normal([1, 3]), name='b3')
        # 定义输出层的weight：weight的维度为[in_size,out_size]
        self.out_w = tfe.Variable(tf.random_normal([3, 1]), name='out_w')
        # 定义输出层的bias：bias的维度固定为[1,out_size]
        self.out_b = tfe.Variable(tf.random_normal([1, 1]), name='out_b')
    def forward(self,inputs):
        '''模型结构为1->4->5->3->1'''
        # 计算第一层输出（具有激活函数）
        layer1 = tf.nn.relu(tf.add(tf.matmul(inputs, self.w1), self.b1))
        # 计算第二层输出（具有激活函数）
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.w2), self.b2))
        # 计算第三层输出（具有激活函数）
        layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, self.w3), self.b3))
        # 计算输出层的输出（没有写激活函数）
        out_layer = tf.add(tf.matmul(layer3, self.out_w), self.out_b)
        return out_layer
model = MLPModel()
'''--------模型end--------'''

'''--------定义loss函数 start--------'''
# 损失函数里面必须包含要计算梯度的权重，不然报错：
#   No trainable variables were accessed while the function was being computed.
def loss_fn(forward, inputs, ys):
    return tf.reduce_mean(tf.reduce_sum(
        tf.square(ys - forward(inputs)), reduction_indices=[1]))
'''--------定义loss end--------'''

'''--------定义优化器与梯度函数 start--------'''
# 使用梯度下降优化神经网络
# 0.00001 为学习率，即learning_rate
optimizer = tf.train.GradientDescentOptimizer(0.00001)
# tfe.implicit_gradients相当于一个python修饰器，可以计算变量的梯度
grad_fn = tfe.implicit_gradients(loss_fn)
'''--------定义优化器与梯度函数 end--------'''

'''--------训练start--------'''
for index, batchData in enumerate(tfe.Iterator(ds)):
    # 计算梯度，grad是当前的梯度数组,本代码对应着8个变量
    grad = grad_fn(model.forward, tf.cast(batchData['x'], dtype=tf.float32), tf.cast(batchData['y'], dtype=tf.float32))
    # 优化权重，反向传播
    optimizer.apply_gradients(grad)
    if index % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception as e:
            print(e)
        # 测试拟合的效果
        prediction = model.forward(tf.cast(x_test, dtype=tf.float32))
        lines = ax.plot(x_test, prediction, 'r-', lw=1)
        # 计算当前的loss，注意：这里类型转换的方式
        loss = loss_fn(model.forward, tf.cast(batchData['x'], dtype=tf.float32), tf.cast(batchData['y'], dtype=tf.float32))
        print(loss)
        plt.pause(0.1)
'''--------训练end--------'''
