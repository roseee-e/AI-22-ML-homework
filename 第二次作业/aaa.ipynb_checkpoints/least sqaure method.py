import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 中文显示
import matplotlib
from matplotlib import rcParams
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取数据
path = "C:\\Users\\Joe\Desktop\\regress_data2.csv"
data = pd.read_csv(path)
cols = data.shape[1]
X_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]
X_data.insert(0, 'Ones', 1)
X = X_data.values
Y = y_data.values

# 使用最小二乘法求解线性回归模型
W = np.linalg.inv(X.T @ X) @ X.T @ Y

# 输出参数矩阵 W
print("参数 W:")
print(W)

# 绘制原始数据点
plt.scatter(X[:, 1], Y, color='blue')

# 绘制拟合的线性回归直线
x_values = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
y_values = W[0] + W[1] * x_values
plt.plot(x_values, y_values, color='red', label='预测值')

plt.xlabel('面积')
plt.ylabel('价格')
plt.title('最小二乘法的面积与房价关系的预测')
plt.legend()
plt.show()
