import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams  ## run command settings for plotting

config = {
    "mathtext.fontset":'stix',
    "font.family":'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,   # 字号，大家自行调节
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)  ## 设置画图的一些参数
# 读取数据
path = 'C:/Users/14774/Downloads/regress_data1.csv'
data = pd.read_csv(path)

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
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 1], y, color='blue', label='实际值')
plt.plot(X[:, 1], y_pred, color='red', label='预测值')
plt.xlabel('人口')
plt.ylabel('收益')
plt.legend()
plt.title('线性回归模型预测结果-无学习率',color='red')
plt.show()
