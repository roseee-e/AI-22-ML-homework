import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)


path = r"C:\Users\邢永远\OneDrive\桌面\regress_data1.csv"
data = pd.read_csv(path)

cols = data.shape[1]
X_data = data.iloc[:, :cols - 1]  # X是所有行，去掉最后一列， 未标准化
y_data = data.iloc[:, cols - 1:]  # X是所有行，最后一列

X_data=(X_data-X_data['人口'].min())/(X_data['人口'].max()-X_data['人口'].min())
X_data.insert(0, 'Ones', 1)

X = X_data.values
Y = y_data.values

W = np.linalg.inv(X.T @ X) @ X.T @ Y

x = np.linspace(X_data['人口'].min(), X_data['人口'].max(), 100)
f = W[0, 0] + (W[1, 0] * x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))  # 创建一个包含两个子图的图形

# 在第一个子图中绘制散点图
ax1.scatter(X_data['人口'], y_data['收益'])
ax1.set_xlabel('人口')
ax1.set_ylabel('收益')
ax1.set_title('数据散点图')

ax2.plot(x, f, 'r', label='预测值')
ax2.scatter(X_data['人口'], y_data['收益'], label='训练数据')
ax2.legend(loc=2)
ax2.set_xlabel('人口')
ax2.set_ylabel('收益')
ax2.set_title('预测收益和人口规模')
plt.show()
