# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import pandas as pd
import numpy as np

# 加载数据集
data = pd.read_csv(r'D:\python数据\regress_data1.csv')
X = data.iloc[:, 0]  # 特征
y = data.iloc[:, 1]  # 目标变量

# 初始化参数
theta0 = 0  # 截距
theta1 = 0  # 斜率
alpha = 0.01  # 学习率
iterations = 1000  # 迭代次数

# 梯度下降算法
m = len(y)
for i in range(iterations):
    y_pred = theta0 + theta1 * X
    error = y_pred - y
    cost = np.sum(error ** 2) / (2 * m)

    gradient0 = np.sum(error) / m
    gradient1 = np.sum(error * X) / m

    theta0 = theta0 - alpha * gradient0
    theta1 = theta1 - alpha * gradient1

    if i % 100 == 0:
        print(f'Iteration {i}, Cost: {cost}')

print(f'Final theta0: {theta0}, Final theta1: {theta1}')