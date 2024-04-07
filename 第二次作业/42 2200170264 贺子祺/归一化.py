import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "D://regress_data1.csv"
data = pd.read_csv(path)
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values

min_X = np.min(X)
max_X = np.max(X)


X_normalized = 2 * (X - min_X) / (max_X - min_X) - 1


X_b_normalized = np.c_[np.ones((data.shape[0], 1)), X_normalized]


theta_best_normalized = np.linalg.inv(X_b_normalized.T.dot(X_b_normalized)).dot(X_b_normalized.T).dot(y)


print("归一化后的最佳拟合参数：", theta_best_normalized)


X_new_normalized = np.linspace(-1, 1, 100).reshape(-1, 1)
X_new_b_normalized = np.c_[np.ones((len(X_new_normalized), 1)), X_new_normalized]  # 添加偏置项
y_predict_normalized = X_new_b_normalized.dot(theta_best_normalized)


plt.plot(X_new_normalized, y_predict_normalized, "r-")
plt.plot(X_normalized, y, "b.")
plt.axis([-1, 1, 0, 25])
plt.xlabel("X (Normalized)")
plt.ylabel("y")
plt.show()
