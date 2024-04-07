import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
# 设置绘图参数
config = {
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["SimHei"],
    "font.size": 10,
    "axes.unicode_minus": False
}
rcParams.update(config)
path = 'D:/regress_data1.csv'
data = pd.read_csv(path)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = np.c_[np.ones(X.shape[0]), X]
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = X.dot(W)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 1], y, color='green', label='实际值')
plt.plot(X[:, 1], y_pred, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.title('线性回归模型预测结果')
plt.show()
