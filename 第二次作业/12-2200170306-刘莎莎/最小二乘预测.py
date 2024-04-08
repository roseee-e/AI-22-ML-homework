# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:07:49 2024

@author: 86182
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取CSV文件
data1 = pd.read_csv('D:\\QQdocument\\regress_data1.csv')

# 确保列名是"人口"和"收益"
x_data = data1['人口'].values.reshape(-1, 1)  # 需要将一维数组转换为二维数组
y_data = data1['收益'].values

# 创建一个二次多项式特征
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x_data)

# 使用线性回归来拟合多项式特征
model = LinearRegression()
model.fit(x_poly, y_data)

# # 输出模型参数
# print(f"系数: {model.coef_}")
# print(f"截距: {model.intercept_}")

# 预测
x_new = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
x_poly_new = poly_features.transform(x_new)
y_pred = model.predict(x_poly_new)

# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']#解决中文显示问题
plt.scatter(x_data, y_data, color='blue', label='实际值')
plt.plot(x_new, y_pred, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('多项式回归拟合曲线')
plt.legend()
plt.show()