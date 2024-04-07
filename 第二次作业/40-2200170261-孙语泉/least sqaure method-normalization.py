import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib import rcParams  # run command settings for plotting

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False     # 处理负号，即-号
}
rcParams.update(config)  # 设置画图的一些参数
# 读取数据
path = 'C:/Users/22682/Desktop/regress_data1.csv'
data = pd.read_csv(path)    # data 是dataframe 的数据类型
cols = data.shape[1]
x_data = data.iloc[:, :cols-1]     # X是所有行，去掉最后一列，未标准化
y_data = data.iloc[:, cols-1:]     # X是所有行，最后一列

x_data = (x_data-x_data['人口'].min())/(x_data['人口'].max()-x_data['人口'].min())

x_data.insert(1, 'Ones', 1)
X = x_data.values
Y = y_data.values
W = np.array([[0.0], [0.0]])  # 初始化W系数矩阵，w 是一个(2,1)矩阵
W += np.linalg.inv(X.T@X)@X.T@Y
# print(W)

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

