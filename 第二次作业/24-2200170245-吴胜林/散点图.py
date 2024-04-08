import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams  ## run command settings for plotting

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,  # 字号，大家自行调节
    'axes.unicode_minus': False  # 处理负号，即-号
}

rcParams.update(config)  ## 设置画图的一些参数
path = "C:\\作业\\上机\\data夹\\regress_data1.csv"
import pandas as pd

data = pd.read_csv(path)  ## data 是dataframe 的数据类型
data.head()  # 返回data中的前几行数据，默认是前5行。

cols = data.shape[1]
X_data = data.iloc[:, :cols - 1]  # X是人口一列
y_data = data.iloc[:, cols - 1:]  # y是收益一列
# 数据归一化

X_1 = (X_data - X_data.min()) / (X_data.max() - X_data.min())
data.plot(kind='scatter', x='人口', y='收益', figsize=(4, 3))  # 利用散点图可视化数据
import matplotlib

plt.xlabel('人口')
plt.ylabel('收益', rotation=90)
plt.show()
