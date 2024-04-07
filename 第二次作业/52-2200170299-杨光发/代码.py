import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv("C:\\new\\regress_data1.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加偏置项
X_train_scaled = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_scaled = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# 定义损失函数
def compute_cost(X, y, theta, lambd):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2) + (lambd / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost

# 梯度下降法学习线性回归模型
def gradient_descent(X, y, theta, alpha, iterations, lambd):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta = theta - (1 / m) * alpha * (X.T.dot(errors) + lambd * np.concatenate(([0], theta[1:])))
        cost_history[i] = compute_cost(X, y, theta, lambd)
    return theta, cost_history

# 初始化参数并运行梯度下降
theta = np.zeros(X_train_scaled.shape[1])
alpha = 0.01
iterations = 1000
lambd = 1
theta_optimized, cost_history = gradient_descent(X_train_scaled, y_train, theta, alpha, iterations, lambd)

# 计算最小二乘法解
theta_least_squares = np.linalg.inv(X_train_scaled.T.dot(X_train_scaled)).dot(X_train_scaled.T).dot(y_train)

# 计算训练和测试损失
train_loss_gd = compute_cost(X_train_scaled, y_train, theta_optimized, lambd)
test_loss_gd = compute_cost(X_test_scaled, y_test, theta_optimized, lambd)
train_loss_ls = compute_cost(X_train_scaled, y_train, theta_least_squares, 0)
test_loss_ls = compute_cost(X_test_scaled, y_test, theta_least_squares, 0)

# 打印训练和测试损失
print("训练损失（梯度下降法）：", train_loss_gd)
print("测试损失（梯度下降法）：", test_loss_gd)
print("训练损失（最小二乘法）：", train_loss_ls)
print("测试损失（最小二乘法）：", test_loss_ls)

# 画出数据散点图
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')
plt.plot(X_train, X_train_scaled.dot(theta_optimized), color='green', label='Gradient Descent')
plt.plot(X_train, X_train_scaled.dot(theta_least_squares), color='orange', label='Least Squares')
plt.title('Data Scatter Plot and Regression Lines')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig('数据归一化I2正则回归效果数据散点图.png')
plt.show()

# 画出训练和测试损失曲线
plt.plot(range(1, iterations + 1), cost_history, color='blue', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('训练和测试损失曲线.png')
plt.show()

