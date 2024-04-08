import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 加载数据
data = pd.read_csv('regress_data.csv')  # 假设数据保存在名为'regress_data.csv'的文件中
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values.reshape(-1, 1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加偏置项
X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# 使用sklearn的Ridge类进行岭回归（带有正则化）
ridge_reg = Ridge(alpha=1.0, solver='sag', fit_intercept=False)
ridge_reg.fit(X_b, y.ravel())

# 获取训练后的参数
theta = ridge_reg.coef_.T
theta = np.insert(theta, 0, ridge_reg.intercept_)

# 预测新数据点
x_new = np.array([[1], [50000]])
x_new_scaled = scaler.transform(x_new)
X_new_b = np.c_[np.ones((x_new_scaled.shape[0], 1)), x_new_scaled]
y_predict_scaled = ridge_reg.predict(X_new_b)
print("Predictions for new data points (with scaling):", y_predict_scaled)