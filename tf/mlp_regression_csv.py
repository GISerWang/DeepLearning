# -*- coding: utf-8 -*-
# 利用多层感知机拟合回归曲线，曲线方程为：y=2*x^2 + 4*x - 0.5
# 输入层仅有一个神经元，输出层仅有一个神经元，即一对一（x->y），还可以运行一对多，多对多，多对一
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
# batch意味着每次在dataSet中提取500条数据
ds = tf.data.Dataset.from_tensor_slices({
    "x": x_train,
    'y': y_train
}).shuffle(buffer_size=123).repeat().batch(500)
# 获取数据集中的迭代器，用于迭代训练数据
iterator = ds.make_one_shot_iterator()
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
'''模型结构为1->4->5->3->1'''
# 定义输入参数和输出参数：xs与ys是占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 定义第一层的weight：weight的维度为[in_size,out_size]
w1 = tf.Variable(tf.random_normal([1, 4]), name="w1")
# 定义第一层的bias:bias的维度固定为[1,out_size]
b1 = tf.Variable(tf.random_normal([1, 4]), name="b1")
# 计算第一层输出（具有激活函数）
layer1 = tf.nn.relu(tf.add(tf.matmul(xs, w1), b1))

# 定义第二层的weight：weight的维度为[in_size,out_size]
w2 = tf.Variable(tf.random_normal([4, 5]), name="w2")
# 定义第二层的bias:bias的维度固定为[1,out_size]
b2 = tf.Variable(tf.random_normal([1, 5]), name="b2")
# 计算第二层输出（具有激活函数）
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))

# 定义第三层的weight：weight的维度为[in_size,out_size]
w3 = tf.Variable(tf.random_normal([5, 3]), name="w3")
# 定义第三层的bias:bias的维度固定为[1,out_size]
b3 = tf.Variable(tf.random_normal([1, 3]), name="b3")
# 计算第三层输出（具有激活函数）
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, w3), b3))

# 定义输出层的weight：weight的维度为[in_size,out_size]
out_w = tf.Variable(tf.random_normal([3, 1]), name="out_w")
# 定义输出层的bias：bias的维度固定为[1,out_size]
out_b = tf.Variable(tf.random_normal([1, 1]), name="out_b")
# 计算输出层的输出（没有写激活函数）
out_layer = tf.add(tf.matmul(layer3, out_w), out_b)
'''--------模型end--------'''

'''--------定义loss start--------'''
# reduction_indices类似axis参数，也就是沿着轴进行运算
# tf.reduce_sum(arr, reduction_indices=[1])：计算arr每一行的和
# tf.reduce_sum(arr)：计算所有每一行和的平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - out_layer), reduction_indices=[1]))
# 使用梯度下降优化神经网络
# 0.00001 为学习率，即learning_rate
train = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
'''--------定义loss end--------'''

'''--------训练start--------'''
with tf.Session() as sess:
    # 初始化模型的变量
    init = tf.global_variables_initializer()
    sess.run(init)
    next_element = iterator.get_next()
    # 迭代优化模型
    for i in range(100000):
        # 获取要训练的数据
        batchData = sess.run(next_element)
        # 训练数据
        sess.run(train, feed_dict={xs: batchData['x'], ys: batchData['y']})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_train, ys: y_train}))
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception as e:
                print(e)
            prediction = sess.run(out_layer, feed_dict={xs: x_test})
            # plot the prediction
            lines = ax.plot(x_test, prediction, 'r-', lw=1)
            plt.pause(0.1)
'''--------训练end--------'''
