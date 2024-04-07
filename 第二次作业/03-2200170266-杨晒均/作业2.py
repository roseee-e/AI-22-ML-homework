
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams  ## run command settings for plotting

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
path = 'C:/Users/杨晒均/Documents/Tencent Files/2164528672/FileRecv/regress_data1.csv'  # 注意斜杠的格式
data = pd.read_csv(path)

cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]

X_data.insert(0, 'Ones', 1)

X = X_data.values
Y = y_data.values
W=np.array([[0.0],[0.0]])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)

def computeCost(X, Y, W):
    Y_hat = np.dot(X, W)
    loss = np.sum((Y_hat - Y)**2) / (2*X.shape[0])
    return loss

def gradientDescent(X, Y, W, alpha):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = X.T @ (Y_hat - Y) / X.shape[0]
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


alpha = 0.003
iters = 10000
loss_his, W = linearRegression(X_scaled, Y_scaled, alpha, iters)

x_scaled = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
x_scaled = scaler.transform(x_scaled.reshape(-1, 1))
f_scaled = W[0, 0] + (W[1, 0] * x_scaled)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(scaler.inverse_transform(x_scaled), scaler.inverse_transform(f_scaled), 'r', label='预测值')
ax.scatter(X_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('归一化后的回归模型-0.003')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(iters), loss_his, 'r')
ax.set_xlabel('迭代次数')
ax.set_ylabel('代价', rotation=0)
ax.set_title('误差以及训练Epoch数-0.003')
plt.show()


