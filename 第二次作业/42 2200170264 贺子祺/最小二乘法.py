import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "D://regress_data1.csv"
data = pd.read_csv(path)
X = data['人口'].values.reshape(-1, 1)
y = data['收益'].values


X_b = np.c_[np.ones((data.shape[0], 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("最佳拟合参数：", theta_best)



X_min, X_max = X.min(), X.max()
X_new = np.linspace(X_min, X_max, 100).reshape(-1, 1)
X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]

y_predict = X_new_b.dot(theta_best)
print("预测结果：", y_predict)


plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([5, 23, 0, 25])
plt.xlabel("X")
plt.ylabel("y")
plt.show()
