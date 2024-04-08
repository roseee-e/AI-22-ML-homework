# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:14:27 2024

@author: 86182
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
  
path = "E:/课程学习/机器学习/regress_data1.csv"  
data = pd.read_csv(path)  
X = data['人口'].values.reshape(-1, 1)  
y = data['收益'].values  
  
min_X = np.min(X)  
max_X = np.max(X)  
  
# 归一化 X  
X_normalized = 2 * (X - min_X) / (max_X - min_X) - 1  
X_b_normalized = np.c_[np.ones((data.shape[0], 1)), X_normalized]  
  
# 计算最佳拟合参数  
theta_best_normalized = np.linalg.inv(X_b_normalized.T.dot(X_b_normalized)).dot(X_b_normalized.T).dot(y)  
  
print("归一化后的最佳拟合参数：", theta_best_normalized)  
  
# 计算新的 X 值范围，与归一化后的 X 相对应  
X_new_min = 2 * (min_X - min_X) / (max_X - min_X) - 1  
X_new_max = 2 * (max_X - min_X) / (max_X - min_X) - 1  
X_new_normalized = np.linspace(X_new_min, X_new_max, 100).reshape(-1, 1)  
  
# 添加偏置项  
X_new_b_normalized = np.c_[np.ones((len(X_new_normalized), 1)), X_new_normalized]  
  
# 预测新的 y 值  
y_predict_normalized = X_new_b_normalized.dot(theta_best_normalized)  
  
# 绘制图形  
plt.plot(X_new_normalized, y_predict_normalized, "r-", label="预测")  
plt.scatter(X_normalized, y, c="b", marker=".", label="数据点")  
plt.axis([X_new_min, X_new_max, min(y) - 1, max(y) + 1])  # 根据数据调整 y 轴范围  
plt.xlabel("X (Normalized)")  
plt.ylabel("y")  
plt.legend()  # 显示图例  
plt.show()