# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:39:13 2024

@author: 86182
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取CSV文件
data1 = pd.read_csv('D:\\QQdocument\\regress_data1.csv')

# 确保列名是"人口"和"收益"
x_data = data1['人口'].values.reshape(-1, 1)
y_data = data1['收益'].values.reshape(-1, 1)  # 也将y_data转换为二维数组

# 数据归一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_data_scaled = scaler_x.fit_transform(x_data)
y_data_scaled = scaler_y.fit_transform(y_data)

# 创建一个二次多项式特征
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x_data_scaled)

# 使用线性回归来拟合多项式特征
model = LinearRegression()
model.fit(x_poly, y_data_scaled.ravel())  # 使用ravel()将二维数组转换为一维数组

# 预测
x_new = np.linspace(x_data_scaled.min(), x_data_scaled.max(), 100).reshape(-1, 1)
x_poly_new = poly_features.transform(x_new)
y_pred_scaled1 = model.predict(x_poly_new)
y_pred_scaled = y_pred_scaled1.reshape(-1,1)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # 还原到原始尺度

# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.scatter(x_data_scaled, y_data_scaled, color='blue', label='归一化后的实际值')
plt.plot(x_new, y_pred_scaled, color='red', label='归一化后的预测值')
plt.xlabel('归一化后的人口')
plt.ylabel('归一化后的收益')
plt.title('多项式回归拟合曲线（归一化后）')
plt.legend()
plt.show()