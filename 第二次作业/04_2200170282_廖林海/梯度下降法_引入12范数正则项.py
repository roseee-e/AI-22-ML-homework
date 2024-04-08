import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import rcParams

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)
path = "D:\\暂时文件\\regress_data1.csv"
data = pd.read_csv(path)
cols = data.shape[1]
x_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.36, random_state=42)


x_train.insert(0, 'Ones', 1)
X = x_train.values
Y = y_train.values
W = np.array([[0.0], [0.0]])


def computeCost(X, Y, W):
    Y_hat = X@W
    loss = np.sum((Y_hat - Y)**2)/(2*X.shape[0]) + 0.0001/2*np.sum(W**2)
    return loss


def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T@(Y_hat-Y)/X.shape[0]
    W += -alpha * dW
    return W


def linearRegression(X, Y, alpha, iters):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha)
    return loss_his, W  


alpha = 0.0001
iters = 10000
loss_his, W = linearRegression(X, Y, alpha, iters)

x = np.linspace(x_data['人口'].min(), x_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

loss_test = []

for i in range(iters):
    loss = computeCost(X, Y, W)
    loss_test.append(loss)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')
plt.show()
print(loss_his)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r',np.arange(iters), loss_test, 'b')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价')
ax.set_title('训练和测试损失曲线')
plt.show()
