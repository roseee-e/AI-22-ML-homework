import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 读取数据
data = pd.read_csv(r"D:\python\data_input_study\regress_data1.csv")

# 分离特征和目标变量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values


scaler = MinMaxScaler()
X_data =scaler.fit_transform(X)
y_data =y

# 添加截距项
X_data = np.column_stack((np.ones(X_data.shape[0]), X_data))

# 利用最小二乘法求解线性回归模型
W_best = np.linalg.inv(X_data.T @ X_data) @ X_data.T @ y_data

# 预测
x = np.linspace(X_data[:, 1].min(), X_data[:, 1].max(), 100)
X_pred = np.column_stack((np.ones(x.shape), x))
y_pred = np.dot(X_pred, W_best)

# 绘制原始数据和拟合直线
plt.figure(figsize=(8, 4))
plt.scatter(X_data[:, 1], y_data, color='blue', label='训练数据')
plt.plot(x, y_pred, color='red', label='拟合曲线')
plt.xlabel('人口')
plt.ylabel('利润')
plt.legend()
plt.title('归一化_最小二乘法拟合训练数据')
plt.show()