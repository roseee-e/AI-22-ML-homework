import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams

# 设置绘图参数
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
data = pd.read_csv('F:\\mycode\\python\\machine learning\\regress_data1.csv')

# 提取特征和目标变量
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 添加偏置项
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# 使用最小二乘法求解线性回归模型
W = np.linalg.inv(X_scaled.T.dot(X_scaled)).dot(X_scaled.T).dot(y)

# 打印训练后的参数
print("训练后的参数：", W)

# 绘制预测值和数据点的散点图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 1], y, label='训练数据')
x_values = np.linspace(0, 1, 100)
y_values = W[0] + W[1] * x_values

plt.plot(x_values, y_values, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('预测值与训练数据(归一化)')
plt.legend()

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# 引入L2范数正则项
lmbda = 0.1
W_reg = np.linalg.inv(X_b.T.dot(X_b) + lmbda * np.eye(X_b.shape[1])).dot(X_b.T).dot(y)

# 打印训练后的参数
print("训练后的参数：", W_reg)

# 绘制预测值和数据点的散点图
plt.subplot(1, 2, 2)
plt.scatter(X, y, label='训练数据')
x_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_b_values = np.c_[np.ones((x_values.shape[0], 1)), x_values]
y_values = x_b_values.dot(W_reg)

plt.plot(x_values, y_values, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('预测值与训练数据（带L2范数正则项）')
plt.legend()
plt.show()
