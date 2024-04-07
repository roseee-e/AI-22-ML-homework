import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 取数据
data = pd.read_csv("C:\\Users\\28645\\Documents\\Tencent Files\\2864506621\\FileRecv\\regress_data1.csv", header=None)
X = data.iloc[:, 0]
y = data.iloc[:, 1]

# 画出数据散点图
plt.scatter(X, y)
plt.xlabel('Population')
plt.ylabel('Income')
plt.title('Scatter plot of Population vs. Income')
plt.show()

# 实现梯度下降线性回归
def compute_cost(X, y, theta):
    m = len(y)
    J = np.sum((X.dot(theta) - y) ** 2) / (2 * m)
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []
    
    for i in range(iterations):
        theta = theta - (alpha/m) * X.T.dot(X.dot(theta) - y)
        J_history.append(compute_cost(X, y, theta))
    
    return theta, J_history

# 在数据上添加偏置列
X = np.c_[np.ones(len(X)), X]

# 初始化参数
theta = np.zeros(2)

# 设置学习率和迭代次数
alpha = 0.01
iterations = 1500

# 运行梯度下降算法
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# 绘制线性模型以及数据
plt.scatter(X[:,1], y, label='Training Data')
plt.plot(X[:,1], X.dot(theta), color='red', label='Linear Regression')
plt.xlabel('Population')
plt.ylabel('Income')
plt.legend()
plt.title('Linear Regression Fit')
plt.show()

# 绘制代价函数
plt.plot(range(1, iterations + 1), J_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()
