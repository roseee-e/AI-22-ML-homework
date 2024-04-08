import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import pandas as pd

# 配置Matplotlib
config = {
    "mathtext.font set": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
plt.rcParams.update(config)

# 读取CSV文件
path = r'D:\regress_data1.csv'
data = pd.read_csv(path)

# 分离特征和目标变量
cols = data.shape[1]
X_data = data.iloc[:, :cols - 1]  # 特征
y_data = data.iloc[:, cols - 1:]  # 目标变量

# 确保X_data有“Ones”列
X_data.insert(0, 'Ones', 1)

# 将DataFrame转换为NumPy数组
X = X_data.values
y = y_data.values.ravel()

# 初始化岭回归模型，并设置正则化参数alpha
# alpha值越大，正则化效果越强
ridge_reg = Ridge(alpha=1.0)

# 拟合模型
ridge_reg.fit(X, y)

# 输出模型的系数和截距
print('系数:', ridge_reg.coef_)
print('截距:', ridge_reg.intercept_)

# 可视化数据
plt.scatter(X[:, 1], y, color='blue', label='实际数据')  # 假设第二个特征是“人口”
plt.plot(X[:, 1], ridge_reg.predict(X), color='red', linewidth=2, label='岭回归线')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('岭回归模型')
plt.legend()
plt.show()

