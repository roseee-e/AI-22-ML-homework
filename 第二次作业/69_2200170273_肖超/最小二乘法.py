import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False     # 处理负号，即-号
}
rcParams.update(config)  # 设置画图参数
# 读取数据
path='/Users/Zhuanz/Downloads/regress_data1.csv'
data = pd.read_csv(path)
cols = data.shape[1]
x_data = data.iloc[:, :cols-1]
y_data = data.iloc[:, cols-1:]




x_data.insert(1, 'Ones', 1)
X = x_data.values
Y = y_data.values
W = np.array([[0.0], [0.0]])  # 初始化W系数矩阵
W += np.linalg.inv(X.T@X)@X.T@Y


x = np.linspace(x_data['人口'].min(), x_data['人口'].max(), 100)
f = W[1, 0] + (W[0, 0] * x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_data['人口'], data['收益'], label='训练数据')
ax.legend(loc=2)
ax.set_xlabel('人口')
ax.set_ylabel('收益', rotation=90)
ax.set_title('预测收益和人口规模')
plt.show()
