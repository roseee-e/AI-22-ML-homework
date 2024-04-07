import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 中文显示
import matplotlib
from matplotlib import rcParams
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
path = "C:\\Users\\Joe\Desktop\\regress_data2.csv"
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
X_data.insert(0, 'Ones', 1)
X = X_data.values
Y = y_data.values

# 计算线性回归
def computeCost(X, Y, W, lambda_val):
    Y_hat = np.dot(X, W)
    loss = (np.sum((Y_hat - Y) ** 2) + lambda_val * np.sum(W ** 2)) / (2 * X.shape[0])
    return loss

def gradientDescent(X, Y, W, alpha, lambda_val):
    num_train = X.shape[0]
    Y_hat = np.dot(X, W)
    dW = (X.T @ (Y_hat - Y) + lambda_val * W) / num_train
    W += -alpha * dW
    return W

def linearRegression(X, Y, alpha, iters, lambda_val):
    loss_his = []
    feature_dim = X.shape[1]
    W = np.zeros((feature_dim, 1))
    for i in range(iters):
        loss = computeCost(X, Y, W, lambda_val)
        loss_his.append(loss)
        W = gradientDescent(X, Y, W, alpha, lambda_val)
    return loss_his, W
# 获取训练得到的参数
alpha = 0.00000001
iters = 10000
lambda_val = 30000000 # 设置正则化参数
loss_his, W = linearRegression(X, Y, alpha, iters, lambda_val)
# 绘制原始数据点
plt.scatter(X[:, 1], Y, color='blue')
# 绘制拟合的线性回归直线
x_values = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
y_values = W[0] + W[1] * x_values
plt.plot(x_values, y_values, color='red', label='预测值')
plt.xlabel('面积')
plt.ylabel('价格')
plt.title(f'引入L2正则项且L2={lambda_val}',)
plt.legend()
plt.show()
