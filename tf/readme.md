# 使用tensorflow静态图构建神经网络

## 1.介绍
### 1.1在本实例中，如果想将代码直接运行需注意以下几点：
* Python版本3.X（本人使用的是Python 3.6）
* numpy版本：1.16.0
* scipy版本：0.19.1
* tensorflow版本：1.12.0
* tensorboard版本：1.12.2
### 1.2 项目说明
* **mlp_regression_csv**：使用tensorflow静态图建造多层感知机（MLP）拟合曲线，从csv文件中读取数据，没有使用归一化层(Batch Normalization)
* **mlp_classification_csv**：使用tensorflow静态图建造多层感知机（MLP）进行多分类任务，从csv文件中读取数据，没有使用归一化层(Batch Normalization)
* **mlp_classification_image**：使用tensorflow静态图建造多层感知机（MLP）进行多分类任务，在线下载数据，没有使用归一化层(Batch Normalization)
* **mlp_classification**：使用tensorflow静态图建造多层感知机（MLP）进行多分类任务，使用高级API，没有使用归一化层(Batch Normalization)
### 1.3 项目注意

* 在数据拟合的过程中，使用relu激活函数，不能拟合小于0的值，因此推荐使用Leaky ReLU等激活器
* 在数据多分类的过程中，注意使用数据归一化（标准化），有利于加快训练的速度，促进模型的收敛