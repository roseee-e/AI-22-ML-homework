import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams  ## run command settings for plotting
config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
# 读取数据
path = 'C:\Users\cycy20\Downloads\regress_data1.csv'
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
# 数据归一化
X_data = (X_data - X_data.mean()) / X_data.std()
y_data = (y_data - y_data.mean()) / y_data.std()
X_data.insert(0, 'Ones', 1)
X = X_data.values
Y = y_data.values
W = np.array([[0.0], [0.0]])
def computeCostWithL2(X, Y, W, reg_lambda):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0]) + reg_lambda * np.sum(W[1:] ** 2) / 2
    return loss
def gradientDescentWithL2(X, Y, W, alpha, reg_lambda):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (X.T @ (Y_hat - Y) + reg_lambda * np.vstack(([[0]], W[1:]))) / X.shape[0]
    W += -alpha * dW
    return W
def linearRegressionWithL2(X, Y, alpha, iters, reg_lambda):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCostWithL2(X, Y, W, reg_lambda)
        loss_his.append(loss)
        W = gradientDescentWithL2(X, Y, W, alpha, reg_lambda)

    return loss_his, W
alpha = 0.0001
iters = 10000
reg_lambda = 0.1
loss_his, W = linearRegressionWithL2(X, Y, alpha, iters, reg_lambda)
def predict(X, W):
    y_pre = np.dot(X, W)
    return y_pre
# 绘制预测结果和训练数据散点图
x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], y_data, label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益')
ax.set_title('预测收益和人口规模')

# 绘制代价函数随迭代次数变化的曲线
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价')
ax.set_title('误差和训练Epoch数')
plt.show()