import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

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
data = pd.read_csv("C:\\Users\\Joe\Desktop\\regress_data2.csv")
X = data[['面积']].values
y = data['价格'].values

# 归一化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 多项式特征转换
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# 使用多项式回归拟合数据
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_scaled, y)

# 绘制原始数据点
plt.scatter(X_scaled, y, color='blue')

# 绘制拟合曲线
X_plot = np.linspace(np.min(X_scaled), np.max(X_scaled), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', label='拟合曲线')

plt.xlabel('归一化后的面积')
plt.ylabel('价格')
plt.title('归一化后的回归拟合')
plt.legend()
plt.show()
