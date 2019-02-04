# DeepLearning

## 1.介绍
### 1.1在本实例中，如果想将代码直接运行需注意以下几点：
* Python版本3.X（本人使用的是Python 3.6）
* numpy版本：1.16.0
* scipy版本：0.19.1
* tensorflow版本：1.12.0
* tensorboard版本：1.12.2
* pytorch版本：0.4.0
* torchvision版本：0.2.1
### 1.2 项目说明
* data：
* pt：使用pytorch动态图构建神经网络(文件夹)
* tf：使用tensorflow静态图构建神经网络(文件夹)
    * mlp_regression_csv：使用tensorflow静态图建造多层感知机（MLP）拟合曲线，从csv文件中读取数据，没有使用归一化层(Batch Normalization)
* tf_eager：使用tensorflow动态图构建神经网络(文件夹)
