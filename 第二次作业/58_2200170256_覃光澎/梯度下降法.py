import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd

config = {
    "mathtext.font set": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

path = r'D:\regress_data1.csv'
data = pd.read_csv(path)
data.head()

cols = data.shape[1]
X_data = data.iloc[:, :cols - 1]  # X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:, cols - 1:]  # X是所有行，最后一列

data.describe()

data.plot(kind='scatter', x='人口', y='收益', figsize=(4, 3))  # 利用散点图可视化数据

plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()

X_data.insert(0, 'Ones', 1)
X_data.head()  # head()是观察前5行

y_data.head()
X = X_data.values
Y = y_data.values
W = np.array([[0.0], [0.0]])

(X.shape, Y.shape, W.shape)


def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])  # (m,n) @ (n, 1) -> (n, 1)
    return loss


def computeCost1(X, Y, W):
    Y_hat = X @ W
    loss = np.sum((Y_hat - Y) ** 2) / (2 * X.shape[0])  # (m,n) @ (n, 1) -> (n, 1)
    return loss


def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / X.shape[0]
    #     dW = X.T@(Y_hat-Y)
    W += -alpha * dW
    return W


def linearRegression(X, Y, alpha, iters):
    loss_his = []
    # step1: initialize the model parameters
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))  ## 初始化W系数矩阵，w 是一个(feature_dim,1)矩阵
    ## repeat step 2 and step 3 untill to the convergence or the end of iterations
    for i in range(iters):
        # step2 : using the initilized parameters to predict the output and calculate the loss
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        # step3: using the gradient decent method to update the parameters
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W  ## 返回损失和模型参数。


def predict(X, W):
    '''
    输入：
        X：测试数据集
        W：模型训练好的参数
    输出：
        y_pre：预测值
    '''
    y_pre = np.dot(X, W)
    return y_pre


alpha = 0.0001
iters = 10000
loss_his, W = linearRegression(X, Y, alpha, iters)

x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
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