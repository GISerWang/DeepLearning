# -*- coding: utf-8 -*-
# 利用多层感知机多分类，测试数据mnist的csv版本(第一列是数据标签，剩下的为784个像素值)
# 输入层具有28*28个神经元，输出层有10个神经元
# 使用tensorflow高级API进行多分类任务
import tensorflow as tf
import numpy as np
'''--------加载数据start--------'''
# 加载训练数据及测试数据
trainData = np.loadtxt('../data/mnist_train.csv',delimiter=",")
testData = np.loadtxt('../data/mnist_test.csv',delimiter=",")
# 提取训练数据的x 及 y
x_train = trainData[:, 1:]/255
y_train = trainData[:, 0]
# 提取测试数据的x,y
x_test = testData[:, 1:]/255
y_test = testData[:, 0]
# 将训练数据及测试数据的label进行one-hot编码
with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, 10))
# 将训练数据封装到dataSet中
# shuffle意味着打乱训练数据集中的顺序
# repeat意味着训练数据无限重复
# batch意味着每次在dataSet中提取256条数据
ds = tf.data.Dataset.from_tensor_slices({
    "x": x_train,
    'labels': y_train
}).shuffle(buffer_size=123).repeat().batch(256)
# 获取数据集中的迭代器，用于迭代训练数据
iterator = ds.make_one_shot_iterator()
'''--------加载数据end--------'''

'''--------模型start--------'''
'''模型结构为784->32->64->32->10'''
# 定义输入参数和输出参数：xs与ys是占位符
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
# 与tf.layers.Dense不同，tf.layers.Dense是类，适合应用于动态图；tf.layers.dense是函数，适合应用于静态图
# 计算第一层输出（具有激活函数），仅需要定义输出单元，不需要定义输入单元
layer1_out = tf.layers.dense(xs, units=32, activation=tf.nn.relu)
# 计算第二层输出（具有激活函数），仅需要定义输出单元，不需要定义输入单元
layer2_out = tf.layers.dense(layer1_out, units=64, activation=tf.nn.relu)
# 计算第三层输出（具有激活函数），仅需要定义输出单元，不需要定义输入单元
layer3_out = tf.layers.dense(layer2_out, units=32, activation=tf.nn.relu)
# 计算输出层的输出（没有写激活函数），仅需要定义输出单元，不需要定义输入单元
out_layer = tf.layers.dense(layer3_out, units=10)
'''--------模型end--------'''

'''--------定义loss start--------'''
# ys与out_layer都是one-hot类型的向量
# tf.nn.softmax_cross_entropy_with_logits_v2等价于tf.nn.softmax+cross_entropy
#       1.最后一层不需要使用激活函数softmax，直接使用softmax_cross_entropy_with_logits_v2即可
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=out_layer))
# 使用AdamOptimizer优化神经网络：收敛速度快
# 使用梯度下降优化神经网络：收敛速度慢
# 0.01 为学习率，即learning_rate
train = tf.train.AdamOptimizer(0.01).minimize(loss)
'''--------定义loss end--------'''

'''--------训练start--------'''
with tf.Session() as sess:
    # 初始化模型的变量
    init = tf.global_variables_initializer()
    sess.run(init)
    next_element = iterator.get_next()
    # 迭代优化模型
    for i in range(1000000):
        # 获取要训练的数据
        batchData = sess.run(next_element)
        # 训练数据，即优化loss
        sess.run(train, feed_dict={xs: batchData['x'], ys: batchData['labels']})
        if i % 50 == 0:
            y_pre = sess.run(out_layer, feed_dict={xs: x_test})
            correct_prediction = tf.equal(tf.argmax(y_pre, 1), y_test)
            # tf.cast 类型转换
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy))
'''--------训练end--------'''
