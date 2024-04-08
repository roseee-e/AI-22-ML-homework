import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('regress_data1.csv')
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values.reshape(-1, 1)

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加偏置项
X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# 初始化参数
theta = np.zeros(X_b.shape[1])


# 定义梯度下降函数，包含正则项
def ridge_regression_gradient_descent(X, y, theta, alpha, iterations, lambda_val):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta)))

    for it in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y
        regularized_term = (lambda_val / m) * theta
        regularized_term[0] = 0  # 不对偏置项应用正则化
        gradient = (1 / m) * X.T.dot(error) + regularized_term
        theta = theta - alpha * gradient
        theta_history[it, :] = theta.T
        cost = compute_cost_with_regularization(X, y, theta, lambda_val)
        cost_history[it] = cost

    return theta, cost_history, theta_history


# 定义包含正则项的成本计算函数
def compute_cost_with_regularization(X, y, theta, lambda_val):
    m = len(y)
    predictions = X.dot(theta)
    sq_error = (predictions - y) ** 2
    regularization = (lambda_val / (2 * m)) * np.sum(theta[1:] ** 2)  # 不包括偏置项
    return (1 / (2 * m)) * np.sum(sq_error) + regularization


# 设置梯度下降参数
alpha = 0.01
iterations = 1000
lambda_val = 1.0  # 正则化强度

# 运行梯度下降
theta, cost_history, theta_history = ridge_regression_gradient_descent(X_b, y, theta, alpha, iterations, lambda_val)

# 预测新数据点
x_new = np.array([[1], [50000]])
x_new_scaled = scaler.transform(x_new)
X_new_b = np.c_[np.ones((x_new_scaled.shape[0], 1)), x_new_scaled]
y_predict = X_new_b.dot(theta)
print("Predictions for new data points:", y_predict)

# 绘制成本变化曲线（可选）
import matplotlib.pyplot as plt

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost over iterations')
plt.show()