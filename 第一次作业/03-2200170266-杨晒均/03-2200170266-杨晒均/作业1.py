
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 设置绘图参数
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = ['SimHei']
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False
# 读取数据
data = pd.read_csv('C:/Users/杨晒均/Documents/Tencent Files/2164528672/FileRecv/regress_data1.csv')
# 提取特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# 添加偏置项
X = np.c_[np.ones(X.shape[0]), X]
# 最小二乘法求解线性回归模型
W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# 预测
y_pred = X.dot(W)
# 绘制结果
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 1], y, color='blue', label='实际值')
plt.plot(X[:, 1], y_pred, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.title('线性回归模型预测结果')
plt.show()

