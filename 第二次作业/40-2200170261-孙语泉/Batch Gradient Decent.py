import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib import rcParams  # run command settings for plotting

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False     # 处理负号，即-号
}
rcParams.update(config)  # 设置画图的一些参数
# 读取数据
path = 'C:/Users/22682/Desktop/regress_data1.csv'
data = pd.read_csv(path)    # data 是dataframe 的数据类型
cols = data.shape[1]
x_data = data.iloc[:, :cols-1]     # X是所有行，去掉最后一列，未标准化
y_data = data.iloc[:, cols-1:]     # X是所有行，最后一列

print(data.describe())    # 查看数据的统计信息

# data.plot(kind='scatter', x='人口', y='收益', figsize=(8, 7))   # 利用散点图可视化数据
# plt.xlabel('人口')
# plt.ylabel('收益', rotation=90)
# plt.show()

x_data.insert(0, 'Ones', 1)
X = x_data.values
Y = y_data.values
W = np.array([[0.0], [0.0]])    # 初始化W系数矩阵，w 是一个(2,1)矩阵

def computeCost(X, Y, W):
    Y_hat = X@W
    loss = np.sum((Y_hat - Y)**2)/(2*X.shape[0])   # (m,n) @ (n, 1) -> (n, 1)
    return loss

def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T@(Y_hat-Y)/X.shape[0]
#     dW = X.T@(Y_hat-Y)
    W += -alpha * dW
    return W

def linearRegression(X,Y, alpha, iters):
    loss_his = []
    # step1: initialize the model parameters
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))   # 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    for i in range(iters):
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W  # 返回损失和模型参数。

alpha = 0.0001
iters = 10000
loss_his, W = linearRegression(X, Y, alpha, iters)

x = np.linspace(x_data['人口'].min(), x_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数')
plt.show()

