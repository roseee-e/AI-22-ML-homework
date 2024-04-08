
import numpy as np
from matplotlib import rcParams, pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 配置matplotlib参数（保持不变）
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimHei'],
    "font.size": 10,
    'axes.unicode_minus': False
}
rcParams.update(config)

# 读取CSV文件（保持不变）
path = r'D:\regress_data1.csv'
data = pd.read_csv(path)

# 分离特征和目标变量（保持不变）
cols = data.shape[1]
X_data = data.iloc[:, :cols - 1]  # 特征
y_data = data.iloc[:, cols - 1:]  # 目标变量

# 确保X_data和y_data是NumPy数组，因为scikit-learn需要这种格式
X = X_data.values
y = y_data.values.ravel()  # ravel()用于将二维数组转换为一维数组

# 初始化线性回归模型
model = LinearRegression()

# 拟合模型，即利用最小二乘法求解线性回归模型
model.fit(X, y)

# 输出模型的系数和截距
print('系数:', model.coef_)
print('截距:', model.intercept_)

# 预测目标变量
y_pred = model.predict(X)

# 计算模型性能指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print('均方误差:', mse)
print('R^2得分:', r2)

# 可视化结果（可选）
plt.scatter(X, y, color='blue', label='实际数据')
plt.plot(X, y_pred, color='red', linewidth=2, label='预测线')
plt.xlabel('特征')
plt.ylabel('目标变量')
plt.title('线性回归模型')
plt.legend()
plt.show()

