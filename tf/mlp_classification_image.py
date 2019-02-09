# -*- coding: utf-8 -*-
# 利用多层感知机多分类，测试数据为在线的minist图像
# 输入层具有28*28个神经元，输出层有10个神经元
# 步骤：
#   1.首先定义holders
#   2.其次定义图结构
#   3.运行网络
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
'''--------加载数据start--------'''
mnist = input_data.read_data_sets("../data/mnist_tensorflow/", one_hot = True)
x_test = mnist.test.images
y_test = mnist.test.labels
'''--------加载数据end--------'''

'''--------模型start--------'''
'''模型结构为784->32->64->32->10'''
# 定义输入参数和输出参数：xs与ys是占位符
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
# 定义第一层的weight：weight的维度为[in_size,out_size]
w1 = tf.Variable(tf.random_normal([784, 32]), name="w1")
# 定义第一层的bias:bias的维度固定为[1,out_size]
b1 = tf.Variable(tf.random_normal([1, 32]), name="b1")
# 计算第一层输出（具有激活函数）
layer1 = tf.nn.relu(tf.add(tf.matmul(xs, w1), b1))

# 定义第二层的weight：weight的维度为[in_size,out_size]
w2 = tf.Variable(tf.random_normal([32, 64]), name="w2")
# 定义第二层的bias:bias的维度固定为[1,out_size]
b2 = tf.Variable(tf.random_normal([1, 64]), name="b2")
# 计算第二层输出（具有激活函数）
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))

# 定义第三层的weight：weight的维度为[in_size,out_size]
w3 = tf.Variable(tf.random_normal([64, 32]), name="w3")
# 定义第三层的bias:bias的维度固定为[1,out_size]
b3 = tf.Variable(tf.random_normal([1, 32]), name="b3")
# 计算第三层输出（具有激活函数）
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, w3), b3))

# 定义输出层的weight：weight的维度为[in_size,out_size]
out_w = tf.Variable(tf.random_normal([32, 10]), name="out_w")
# 定义输出层的bias：bias的维度固定为[1,out_size]
out_b = tf.Variable(tf.random_normal([1, 10]), name="out_b")
# 计算输出层的输出（没有写激活函数）
out_layer = tf.add(tf.matmul(layer3, out_w), out_b)
'''--------模型end--------'''

'''--------定义loss start--------'''
# ys与out_layer都是one-hot类型的向量

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=out_layer))
# 使用梯度下降优化神经网络
# 0.1 为学习率，即learning_rate
train = tf.train.AdamOptimizer(0.01).minimize(loss)
'''--------定义loss loss--------'''

'''--------训练start--------'''
with tf.Session() as sess:
    # 初始化模型的变量
    init = tf.global_variables_initializer()
    sess.run(init)
    # 迭代优化模型
    for i in range(1000000):
        # 获取要训练的数据
        batch_xs, batch_ys = mnist.train.next_batch(1000)
        # 训练模型，即优化loss
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            # 测试数据
            y_pre = sess.run(out_layer, feed_dict={xs: x_test})
            # 在测试数据的时候，由于是分类任务，在测试时，不需要softmax，直接argmax即可。
            # 判断有几个数据分类正确，np.argmax(y_test, 1)按行查找，最大的索引
            correct_prediction = np.equal(np.argmax(y_pre, 1), np.argmax(y_test, 1))
            # 计算正确率
            # astype将True转为1，False转为0，从而使用np.mean计算平均数
            accuracy = np.mean(correct_prediction.astype(np.float))
            print(accuracy)
'''--------训练end--------'''
