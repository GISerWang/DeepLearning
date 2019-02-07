# -*- coding: utf-8 -*-
# 利用多层感知机多分类，测试数据为在线的minist图像(eager模式，即tensorflow动态图机制)
# 输入层具有28*28个神经元，输出层有10个神经元
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 启动tensorflow的动态图机制
tfe.enable_eager_execution()
'''--------加载数据start--------'''
mnist = input_data.read_data_sets("../data/mnist_tensorflow/", one_hot = True)
x_test = mnist.test.images
y_test = mnist.test.labels
'''--------加载数据end--------'''

'''--------模型start--------'''
class MLPModel(object):
    def __init__(self):
        # 定义第一层的weight：weight的维度为[in_size,out_size]
        self.w1 = tfe.Variable(tf.random_normal([784, 32]), name="w1")
        # 定义第一层的bias:bias的维度固定为[1,out_size]
        self.b1 = tfe.Variable(tf.random_normal([1, 32]), name="b1")
        # 定义第二层的weight：weight的维度为[in_size,out_size]
        self.w2 = tfe.Variable(tf.random_normal([32, 64]), name="w2")
        # 定义第二层的bias:bias的维度固定为[1,out_size]
        self.b2 = tfe.Variable(tf.random_normal([1, 64]), name="b2")
        # 定义第三层的weight：weight的维度为[in_size,out_size]
        self.w3 = tfe.Variable(tf.random_normal([64, 32]), name="w3")
        # 定义第三层的bias:bias的维度固定为[1,out_size]
        self.b3 = tfe.Variable(tf.random_normal([1, 32]), name="b3")
        # 定义输出层的weight：weight的维度为[in_size,out_size]
        self.out_w = tfe.Variable(tf.random_normal([32, 10]), name="out_w")
        # 定义输出层的bias：bias的维度固定为[1,out_size]
        self.out_b = tfe.Variable(tf.random_normal([1, 10]), name="out_b")

    def forward(self, inputs):
        '''模型结构为784->32->64->32->10'''
        # 计算第一层输出（具有激活函数）
        layer1 = tf.nn.relu(tf.add(tf.matmul(inputs, self.w1), self.b1))
        # 计算第二层输出（具有激活函数）
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.w2), self.b2))
        # 计算第三层输出（具有激活函数）
        layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, self.w3), self.b3))
        # 计算输出层的输出（没有写激活函数）
        out_layer = tf.add(tf.matmul(layer3, self.out_w), self.out_b)
        return out_layer
'''--------模型end--------'''

'''--------定义loss函数 start--------'''
# 必须在loss_fn函数里面进行前项传播，然后经implicit_gradients包装后，才能进行梯度计算
def loss_fn(forward, inputs, ys):
    # tf.nn.softmax_cross_entropy_with_logits_v2等价于tf.nn.softmax+cross_entropy
    #       1.最后一层不需要使用激活函数softmax，直接使用softmax_cross_entropy_with_logits_v2即可
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=forward(inputs)))
'''--------定义loss end--------'''
# ys与out_layer都是one-hot类型的向量
optimizer = tf.train.AdamOptimizer(0.01)
grad_fn = tfe.implicit_gradients(loss_fn)
'''--------定义loss loss--------'''

'''--------训练start--------'''
model = MLPModel()
for i in range(100000):
    # 获取要训练的数据
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    # 计算每一个变量的梯度
    grad = grad_fn(model.forward, batch_xs, batch_ys)
    # 优化梯度
    optimizer.apply_gradients(grad)
    if i % 50 == 0:
        y_pre = model.forward(tf.cast(x_test, dtype=tf.float32))
        correct_prediction = np.equal(np.argmax(y_pre.numpy(), 1), np.argmax(y_test, 1))
        accuracy = np.mean(correct_prediction.astype(np.float))
        print(accuracy)
'''--------训练end--------'''
