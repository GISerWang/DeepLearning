# -*- coding: utf-8 -*-
# 利用卷积神经网络多分类，测试数据mnist的csv版本
# 使用tensorflow高级API进行构建卷积神经网络
# 输入数据具有四个维度，一般为[batch,height,width,channels]，输出层有10个神经元
import tensorflow as tf
import numpy as np
'''--------加载数据start--------'''
# 加载训练数据及测试数据
trainData = np.loadtxt('../data/mnist_train.csv', delimiter=",")
testData = np.loadtxt('../data/mnist_test.csv', delimiter=",")
# 提取训练数据的x 及 y
# 将训练数据转换为相应的形状[batch,height,width,channels]
x_train = trainData[:, 1:].reshape(-1, 28, 28, 1)
y_train = trainData[:, 0]
# 提取测试数据的x,y
# 将测试数据转换为相应的形状[batch,height,width,channels]
x_test = testData[:, 1:].reshape(-1, 28, 28, 1)
y_test = testData[:, 0]

# 将训练数据及测试数据的label进行one-hot编码
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, 10))
# 将训练数据封装到dataSet中
# shuffle意味着打乱训练数据集中的顺序
# repeat意味着训练数据无限重复
# batch意味着每次在dataSet中提取500条数据
ds = tf.data.Dataset.from_tensor_slices({
    "x": x_train,
    'labels': y_train
}).shuffle(buffer_size=123).repeat().batch(32)
# 获取数据集中的迭代器，用于迭代训练数据
iterator = ds.make_one_shot_iterator()
'''--------加载数据end--------'''
'''--------模型start--------'''
# 定义输入参数和输出参数：xs与ys是占位符
xs = tf.placeholder(tf.float32, [None, 28, 28, 1])
ys = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)
# 卷积+归一化+drop+池化+卷积+归一化+drop+池化+全连接层+归一化+drop+全连接层即：
# conv->bn->drop->pool->conv->bn->drop->pool->fc->bn->drop->fc

# 进行卷积操作
# strides = [stride, stride]:stride是卷积核的移动步长
# padding可以是SAME也可以是VALID
# SAME表示填充，填充的大小与卷积核的大小的关系为：2n+1=卷积核大小，n为填充大小
# kernel_size定义卷积核的大小，卷积核的长为5，宽为5，
# filters卷积核的个数为32个
# 输入图像的波段被省略，即channels被省略
# 输入的矩阵为：[32, 28, 28, 1],输出的矩阵为：[32, 28, 28, 32]
conv1 = tf.layers.conv2d(xs, filters=32, kernel_size=5, strides=(1, 1), padding='SAME')
# 进行归一化操作，归一化操作定义在非线性激励之前
# momentum为滑动平均算法的超参数
conv1 = tf.layers.batch_normalization(conv1, momentum=0.9, training=is_training)
# 进行非线性激励
conv1 = tf.nn.relu(conv1)
# 进行dropout操作,既有20%的几率drop掉神经元
conv1 = tf.layers.dropout(conv1, rate=0.2, training=is_training)
# 进行池化操作
# pool_size的大小，为池化窗口为3
# strides=[stride,stride],stride是池化窗口的移动步长
# 设移动步长为k，那么featureMap的宽和高变为原来的1/k，
# 输入的矩阵为：[32, 28, 28, 32],输出的矩阵为：[32, 14, 14, 32]
pool2 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=(2, 2), padding='SAME')
# 进行卷积操作
# strides = [stride, stride]
# stride是卷积核的移动步长
# padding可以是SAME也可以是VALID
# SAME表示填充，填充的大小与卷积核的大小的关系为：2n+1=卷积核大小，n为填充大小
# kernel_size定义卷积核的大小，卷积核的长为5，宽为5，卷积核的个数为64个
# 输入图像的波段为32，但是被省略了
# 输入的矩阵为：[32, 14, 14, 32],输出的矩阵为：[32, 14, 14, 64]
conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=5, strides=(1, 1), padding='SAME')
# 进行归一化操作，归一化操作定义在非线性激励之前
# momentum为滑动平均算法的超参数
conv3 = tf.layers.batch_normalization(conv3, momentum=0.9, training=is_training)
# 进行非线性激励
conv3 = tf.nn.relu(conv3)
# 进行dropout操作，既有20%的几率drop掉神经元
conv3 = tf.layers.dropout(conv3, rate=0.2, training=is_training)

# 进行池化操作
# pool_size的大小，为池化窗口为3
# strides：stride是池化窗口的移动步长
# 设移动步长为k，那么卷积核的宽和高变为原来的1/k，
# 输入的矩阵为：[32, 14, 14, 64],输出的矩阵为：[32, 7, 7, 64]
pool4 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=(2, 2), padding='SAME')
# 进行全连接矩阵乘法
# 定义第五层的权重，权重的形状为[in,out]
# 其中in等于上一层特征图，经拉直后的大小即：[32, 7, 7, 64]->[32, 7*7*64]
# out为下一层全连接层的个数
# bias的大小与out相等
pool4 = tf.reshape(pool4, [-1, 7*7*64])
fc5 = tf.layers.dense(pool4, 1000)
# 进行归一化操作，归一化操作定义在非线性激励之前
# momentum为滑动平均算法的超参数
fc5 = tf.layers.batch_normalization(fc5, momentum=0.9, training=is_training)
# 定义第五层的非线性激励
fc5 = tf.nn.relu(fc5)
# 进行dropout操作，既有20%的几率drop掉神经元
fc5 = tf.layers.dropout(fc5, rate=0.2, training=is_training)
# 最后一层：全连接层
# 定义最后一层的权重，权重的形状为[in,out]
# 其中in等于out，即：1000
# out为最后一层分类的个数
# bias的大小与out相等
out = tf.layers.dense(fc5, 10)
'''--------模型end--------'''

'''--------定义loss start--------'''
# ys与out_layer都是one-hot类型的向量

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=out))
# 使用梯度下降优化神经网络
# 0.1 为学习率，即learning_rate
# 获得模型中需要更新的量，由于我们使用了归一化层，滑动均值与滑动方差需要每次求
# 只要在静态框架使用tf.layers.batch_normalization，都必须要这么做
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer(0.01).minimize(loss)
'''--------定义loss loss--------'''

'''--------训练start--------'''
with tf.Session() as sess:
    # 初始化模型的变量
    init = tf.global_variables_initializer()
    sess.run(init)
    next_element = iterator.get_next()
    for i in range(1000000):
        # 获取要训练的数据
        batchData = sess.run(next_element)
        # 训练模型，即优化loss
        sess.run(train, feed_dict={xs: batchData['x'],
                                   ys: batchData['labels'],
                                   is_training: True})
        if i % 50 == 0:
            # 测试数据
            y_pre = sess.run(out, feed_dict={xs: x_test, is_training: False})
            # 在测试数据的时候，由于是分类任务，在测试时，不需要softmax，直接argmax即可。
            # 判断有几个数据分类正确，tf.argmax(y_test, 1)按行查找，最大的索引
            correct_prediction = tf.equal(tf.argmax(y_pre, 1), y_test)
            # 计算正确率
            # tf.cast() 将True转为1，False转为0，从而使用reduce_mean计算平均数
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy))
'''--------训练end--------'''
