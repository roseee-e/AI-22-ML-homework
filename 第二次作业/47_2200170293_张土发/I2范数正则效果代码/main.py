# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import pandas as pd
import numpy as np

# 加载数据集
data = pd.read_csv(r'D:\python数据\regress_data1.csv')
X = data.iloc[:, 0]  # 特征
y = data.iloc[:, 1]  # 目标变量

# 添加一列全为1的特征，用于计算截距项
X = np.c_[np.ones(X.shape[0]), X]

# 初始化参数
theta = np.zeros(X.shape[1])  # 参数向量
alpha = 0.01  # 学习率
lmbda = 0.1  # 正则化参数
iterations = 1000  # 迭代次数

# 梯度下降算法
m = len(y)
for i in range(iterations):
    y_pred = np.dot(X, theta)
    error = y_pred - y
    cost = np.sum(error ** 2) / (2 * m) + lmbda * np.sum(theta[1:] ** 2) / (2 * m)  # 添加L2正则项

    gradient = np.dot(X.T, error) / m
    gradient[1:] += lmbda * theta[1:] / m  # 更新梯度，对除了截距项之外的参数应用正则化

    theta = theta - alpha * gradient

    if i % 100 == 0:
        print(f'Iteration {i}, Cost: {cost}')

print(f'Final theta: {theta}')