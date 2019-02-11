# -*- coding: utf-8 -*-
# 利用卷积神经网络多分类，测试数据为在线的minist图像(eager模式，即tensorflow动态图机制)
# 使用最原生的方法，自己初始化权重W和bias
# 输入数据具有四个维度，一般为[batch,height,width,channels]，输出层有10个神经元
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 启动tensorflow的动态图机制
tfe.enable_eager_execution()
'''--------加载数据start--------'''
mnist = input_data.read_data_sets("../data/mnist_tensorflow/", one_hot=True)
x_test = mnist.test.images.reshape(-1, 28, 28, 1)
y_test = mnist.test.labels
'''--------加载数据end--------'''
'''--------模型start--------'''
# 卷积+归一化+drop+池化+卷积+归一化+drop+池化+全连接层+归一化+drop+全连接层即：
# conv->bn->drop->pool->conv->bn->drop->pool->fc->bn->drop->fc
class CNNModel(object):
    def __init__(self):
        # 创建第一层卷积层的权重
        # 定义卷积核的大小，卷积核的长为5，宽为5，输入图像的波段为1，卷积核的个数为32个
        self.conv1_w = tfe.Variable(tf.random_normal([5, 5, 1, 32]), name="conv1_w")
        # 定义第一层卷积核的偏重bias，每一个卷积核具有一个bias，由于具有32个卷积核，因此具有32个bias
        self.conv1_b = tfe.Variable(tf.random_normal([32]))
        # 定义第一层卷积层后面的归一化层滑动平均值
        # 滑动平均值用于测试阶段
        self.mean1 = tfe.Variable(tf.zeros([32]))
        self.var1 = tfe.Variable(tf.ones([32]))
        # 定义第一层卷积层后面的归一化层权重，权重的个数与特征数目相等（即：卷积核个数）
        # 默认放大系数为1，偏移系数为0
        self.bt_offset1 = tfe.Variable(tf.zeros([32]))
        self.bt_scale1 = tfe.Variable(tf.ones([32]))
        # 创建第三层的卷积层的卷积核
        # 定义卷积核的大小，卷积核的长为5，宽为5，输入图像的波段为32，卷积核的个数为64个
        self.conv3_w = tfe.Variable(tf.random_normal([5, 5, 32, 64]), name="conv3_w")
        # 定义第三层卷积核的偏重bias，每一个卷积核具有一个bias，由于具有32个卷积核，因此具有32个bias
        self.conv3_b = tfe.Variable(tf.random_normal([64]))
        # 定义第三层卷积层后面的归一化层滑动平均值（默认，均值为0，方差为1）
        # 滑动平均值用于测试阶段
        self.mean2 = tfe.Variable(tf.zeros([64]))
        self.var2 = tfe.Variable(tf.ones([64]))
        # 定义第三层卷积层后面的归一化层权重，权重的个数与特征数目相等（即：卷积核个数）
        # 默认放大系数为1，偏移系数为0
        self.bt_offset2 = tfe.Variable(tf.zeros([64]))
        self.bt_scale2 = tfe.Variable(tf.ones([64]))
        # 定义第五层的全连接层的权重，权重的形状为[in,out]
        # 其中in等于上一层特征图，经拉直后的大小即：[32, 7, 7, 64]->[32, 7*7*64]
        # out为下一层全连接层的个数
        # bias的大小与out相等
        self.fc_w = tfe.Variable(tf.random_normal([7 * 7 * 64, 1000]), name="fc_w")
        self.fc_b = tfe.Variable(tf.random_normal([1000]), name="fc_b")
        # 定义第五层全连接层层后面的归一化层滑动平均值（默认，均值为0，方差为1）
        # 滑动平均值用于测试阶段
        self.mean3 = tfe.Variable(tf.zeros([1000]))
        self.var3 = tfe.Variable(tf.ones([1000]))
        # 定义第三层卷积层后面的归一化层权重，权重的个数与特征数目相等（即：隐藏层的个数）
        # 默认放大系数为1，偏移系数为0
        self.bt_offset3 = tfe.Variable(tf.zeros([1000]))
        self.bt_scale3 = tfe.Variable(tf.ones([1000]))
        self.out_w = tfe.Variable(tf.random_normal([1000, 10]), name="out_w")
        self.out_b = tfe.Variable(tf.random_normal([10]), name="out_b")
    # 定义数据的前馈传播
    #   inputs：需要训练的数据集，用于优化神经网络
    #   keep_prob：用于drop层的，假设为0.1，表示0.1的概率保留神经元
    #   is_training：用于训练，控制归一化层的参数改变
    #   momentum：用于归一化层，滑动平均算法的超参数
    def forward(self, inputs, keep_prob, is_training, momentum=0.9):
        # 卷积操作
        # strides：第一个参数和最后一个参数固定为1，第二个和第三个参数stride是卷积核的移动步长
        # padding可以是SAME也可以是VALID
        # SAME表示填充，填充的大小与卷积核的大小的关系为：2n+1=卷积核大小，n为填充大小
        # 输入的矩阵为：[32, 28, 28, 1],输出的矩阵为：[32, 28, 28, 32]
        conv1 = tf.nn.conv2d(inputs, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME')
        # 计算偏置
        conv1 = tf.nn.bias_add(conv1, self.conv1_b)
        if is_training:
            # 如果是训练模式，计算每个特征在batch下的均值与方差
            # 由于我们的数据格式为：[batch,height,width,channels],因此是axes=[0, 1, 2]
            # 如果我们的格式为：[batch,channels,width,height],axes需要修改为=[0,2,3]
            # 归一化层处于非线性激励之前
            bt_mean1, bt_var1 = tf.nn.moments(conv1, axes=[0, 1, 2])
            # 将数据进行归一化操作（训练模式下，使用批处理下的均值与方差）
            conv1 = tf.nn.batch_normalization(conv1, bt_mean1, bt_var1,
                                              offset=self.bt_offset1,
                                              scale=self.bt_scale1,
                                              variance_epsilon=0.0001)
            # 滑动平均算法，求全局模式下的均值与方差
            # 归一化层处于非线性激励之前
            self.mean1 = bt_mean1 * momentum + self.mean1 * (1-momentum)
            self.var1 = bt_var1 * momentum + self.var1 * (1-momentum)
        else:
            # 测试模式，使用滑动平均求得的均值与方差
            # 归一化层处于非线性激励之前
            conv1 = tf.nn.batch_normalization(conv1, self.mean1, self.var1,
                                              offset=self.bt_offset1,
                                              scale=self.bt_scale1,
                                              variance_epsilon=0.0001)
        # 非线性激励
        conv1 = tf.nn.relu(conv1)
        # 使用drop操作
        conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)
        # 创建第二层：池化层
        # ksize的第一个参数和最后一个参数固定为1，第二个和第三个参数stride是池化核的宽和高
        # strides第一个参数和最后一个参数固定为1，第二个和第三个参数stride是卷积核的移动步长
        # 设移动步长为k，那么卷积核的宽和高变为原来的1/k，
        # 输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 14, 14, 32]
        pool2 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
        # 创建第三层：卷积层
        # strides = [1, stride, stride, 1]
        # 第一个参数和最后一个参数固定为1，第二个和第三个参数stride是卷积核的移动步长
        # padding可以是SAME也可以是VALID
        # SAME表示填充，填充的大小与卷积核的大小的关系为：2n+1=卷积核大小，n为填充大小
        # 定义卷积核的大小，卷积核的长为5，宽为5，输入图像的波段为32，卷积核的个数为64个
        # 输入的矩阵为：[32, 14, 14, 32],输出的矩阵为：[32, 14, 14, 64]
        conv3 = tf.nn.conv2d(pool2,self.conv3_w,strides = [1, 1, 1, 1], padding = 'SAME')
        # 计算偏置
        conv3 = tf.nn.bias_add(conv3, self.conv3_b)
        if is_training:
            # 如果是训练模式，计算每个特征在batch下的均值与方差
            # 由于我们的数据格式为：[batch,height,width,channels],因此是axes=[0, 1, 2]
            # 如果我们的格式为：[batch,channels,width,height],axes需要修改为=[0,2,3]
            # 归一化层处于非线性激励之前
            bt_mean2, bt_var2 = tf.nn.moments(conv3, axes=[0, 1, 2])
            conv3 = tf.nn.batch_normalization(conv3, bt_mean2, bt_var2,
                                              offset=self.bt_offset2,
                                              scale=self.bt_scale2,
                                              variance_epsilon=0.0001)
            # 滑动平均算法，求全局模式下的均值与方差
            # 归一化层处于非线性激励之前
            self.mean2 = bt_mean2 * momentum + self.mean2 * (1-momentum)
            self.var2 = bt_var2 * momentum + self.mean2 * (1-momentum)
        else:
            # 测试模式，使用滑动平均求得的均值与方差
            # 归一化层处于非线性激励之前
            conv3 = tf.nn.batch_normalization(conv3, self.mean2, self.var2,
                                              offset=self.bt_offset2,
                                              scale=self.bt_scale2,
                                              variance_epsilon=0.0001)
        # 定义第三层的卷积核的非线性激励
        conv3 = tf.nn.relu(conv3)
        # 使用drop操作
        conv3 = tf.nn.dropout(conv3, keep_prob=keep_prob)
        # 创建第四层：池化层
        # ksize的第一个参数和最后一个参数固定为1，第二个和第三个参数stride是池化核的宽和高
        # strides第一个参数和最后一个参数固定为1，第二个和第三个参数stride是卷积核的移动步长
        # 设移动步长为k，那么卷积核的宽和高变为原来的1/k，
        # 输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 7, 7, 64]
        pool4 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
        # 定义第五层：全连接层
        # 输入的矩阵为：[32, 14, 14, 64],经reshape后形状为[32, 7*7*64]
        pool4 = tf.reshape(pool4, [-1, 7 * 7 * 64])
        # 全连接计算，输入矩阵为：[32, 7*7*64]，输出矩阵为[32, 1000]
        fc5 = tf.add(tf.matmul(pool4, self.fc_w), self.fc_b)
        if is_training:
            # 如果是训练模式，计算每个特征在batch下的均值与方差
            # 由于我们的数据格式为：[batch,channels],因此是axes=[0]
            # 归一化层处于非线性激励之前
            bt_mean3, bt_var3 = tf.nn.moments(fc5, axes=[0])
            fc5 = tf.nn.batch_normalization(fc5, bt_mean3, bt_var3,
                                              offset=self.bt_offset3,
                                              scale=self.bt_scale3,
                                              variance_epsilon=0.0001)
            self.mean3 = bt_mean3 * momentum + self.mean3 * (1-momentum)
            self.var3 = bt_var3 * momentum + self.mean3 * (1-momentum)
        else:
            # 测试模式，使用滑动平均求得的均值与方差
            # 归一化层处于非线性激励之前
            fc5 = tf.nn.batch_normalization(fc5, self.mean3, self.var3,
                                              offset=self.bt_offset3,
                                              scale=self.bt_scale3,
                                              variance_epsilon=0.0001)
        # 定义第五层的非线性激励
        fc5 = tf.nn.relu(fc5)
        fc5 = tf.nn.dropout(fc5, keep_prob=keep_prob)
        # 最后一层：全连接层
        # 定义最后一层的权重，权重的形状为[in,out]
        # 其中in等于out，即：1000
        # out为最后一层分类的个数
        # bias的大小与out相等
        out = tf.add(tf.matmul(fc5, self.out_w), self.out_b)
        return out
'''--------模型end--------'''

'''--------定义loss函数 start--------'''
# 必须在loss_fn函数里面进行前项传播，然后经implicit_gradients包装后，才能进行梯度计算
def loss_fn(forward, inputs, ys, keep_prob, is_training, momentum=0.9):
    # tf.nn.softmax_cross_entropy_with_logits_v2等价于tf.nn.softmax+cross_entropy
    #       1.最后一层不需要使用激活函数softmax，直接使用softmax_cross_entropy_with_logits_v2即可
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=forward(inputs,keep_prob,is_training,momentum)))
'''--------定义loss函数 end--------'''

'''--------定义优化器与梯度函数 start--------'''
# ys与out_layer都是one-hot类型的向量
optimizer = tf.train.AdamOptimizer(0.01)
grad_fn = tfe.implicit_gradients(loss_fn)
'''--------定义优化器与梯度函数 end--------'''

'''--------训练start--------'''
model = CNNModel()
for i in range(100000):
    # 获取要训练的数据
    batch_xs, batch_ys = mnist.train.next_batch(32)
    # 计算每一个变量的梯度
    grad = grad_fn(model.forward, batch_xs.reshape(-1, 28, 28, 1), batch_ys, 0.8, True)
    # 优化梯度
    optimizer.apply_gradients(grad)
    if i % 50 == 0:
        # 测试阶段
        y_pre = model.forward(tf.cast(x_test, dtype=tf.float32), 1, False)
        # 测试正确率
        correct_prediction = np.equal(np.argmax(y_pre.numpy(), 1), np.argmax(y_test, 1))
        accuracy = np.mean(correct_prediction.astype(np.float))
        print(accuracy)
'''--------训练end--------'''
