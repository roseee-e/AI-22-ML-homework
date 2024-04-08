# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv("D:/python数据/regress_data1.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加偏置项
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 定义梯度下降函数
def gradient_descent(X, y, theta, alpha, iterations, lambda_val):
    m = len(y)
    loss_history_train = []
    loss_history_test = []

    for i in range(iterations):
        # 计算预测值
        y_pred = np.dot(X, theta)

        # 计算损失函数
        loss = np.sum((y_pred - y) ** 2) / (2 * m) + lambda_val * np.linalg.norm(theta[1:]) ** 2

        # 计算梯度
        gradient = np.dot(X.T, (y_pred - y)) / m + 2 * lambda_val * np.r_[0, theta[1:]]

        # 更新参数
        theta = theta - alpha * gradient

        # 计算训练集和测试集上的损失
        y_pred_train = np.dot(X_train, theta)
        loss_train = np.sum((y_pred_train - y_train) ** 2) / (2 * len(y_train))
        loss_history_train.append(loss_train)

        y_pred_test = np.dot(X_test, theta)
        loss_test = np.sum((y_pred_test - y_test) ** 2) / (2 * len(y_test))
        loss_history_test.append(loss_test)

    return theta, loss_history_train, loss_history_test


# 初始化参数
theta = np.zeros(X_train.shape[1])

# 设置超参数
alpha = 0.01
iterations = 1000
lambda_val = 0.1

# 运行梯度下降
theta_gd, loss_history_train, loss_history_test = gradient_descent(X_train, y_train, theta, alpha, iterations,
                                                                   lambda_val)

# 最小二乘法求解线性回归模型
theta_ls = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))

# 绘制训练和测试损失曲线
plt.figure()
plt.plot(range(iterations), loss_history_train, label='Training Loss')
plt.plot(range(iterations), loss_history_test, label='Testing Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Curve')
plt.legend()
plt.show()