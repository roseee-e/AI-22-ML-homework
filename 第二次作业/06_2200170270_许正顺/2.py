import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams

config = {
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["SimHei"],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)
path = 'D:/qq/regress_data1.csv'
data = pd.read_csv(path)
cols = data.shape[1]
X = data.iloc[:, :cols-1]
y = data.iloc[:, cols-1:]
X.insert(0, 'Ones', 1)
X = X.values
y = y.values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)
def computeCost(X, y, W):
    y_pred = X.dot(W)
    loss = np.sum((y_pred - y)**2) / (2*X.shape[0])
    return loss
def gradientDescent(X, y, W, alpha):
    y_pred = X.dot(W)
    dW = X.T @ (y_pred - y) / X.shape[0]
    W -= alpha * dW
    return W
def linearRegression(X, y, alpha, iters):
    loss_history = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, y, W)
        loss_history.append(loss)
        W = gradientDescent(X, y, W, alpha)
    return loss_history, W
alpha = 0.003
iters = 10000
loss_history, W = linearRegression(X_scaled, y_scaled, alpha, iters)
x_scaled = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x_scaled = scaler.transform(x_scaled.reshape(-1, 1))
y_pred_scaled = W[0, 0] + (W[1, 0] * x_scaled)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(scaler.inverse_transform(x_scaled), scaler.inverse_transform(y_pred_scaled), 'r', label='预测值')
ax.scatter(X[:, 1], y, label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('归一化后的回归模型（alpha=0.003）')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_history, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差和训练Epoch数（alpha=0.003）')
plt.show()