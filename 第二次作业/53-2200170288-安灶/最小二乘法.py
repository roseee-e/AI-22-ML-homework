import sys
sys.path.append("D:\python\lib\site-packages")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
path = 'C:\\Users\\86178\\Documents\\Tencent Files\\3061593833\\FileRecv\\regress_data1.csv'
data = pd.read_csv(path)
X_data = data.iloc[:, :-1]  # 特征数据
y_data = data.iloc[:, -1]   # 目标数据

# 添加偏置列
X_data.insert(0, 'Ones', 1)

# 转换为numpy数组
X = X_data.values
y = y_data.values.reshape(-1, 1)

# 最小二乘法求解线性回归模型
theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# 预测值
y_pred = X.dot(theta_best)

# 绘制数据散点图和拟合直线
plt.scatter(X_data['人口'], y_data)
plt.plot(X_data['人口'], y_pred, color='red')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('预测收益和人口规模(最小二乘法)')
plt.show()