import numpy as np
import pandas as pd

# 加载数据
data = pd.read_csv('regress_data1.csv')
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values.reshape(-1, 1)

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# 利用最小二乘法求解参数
# 正规方程为：theta = (X_b.T * X_b).I * X_b.T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 预测新数据点
x_new = np.array([[1], [50000]])
X_new_b = np.c_[np.ones((x_new.shape[0], 1)), x_new]
y_predict = X_new_b.dot(theta_best)
print("Predictions for new data points:", y_predict)